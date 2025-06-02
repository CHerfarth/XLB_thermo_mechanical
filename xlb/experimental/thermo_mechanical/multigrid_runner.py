import xlb
import sys
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
import xlb.experimental
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.velocity_set
import warp as wp
import numpy as np
from typing import Any
import sympy
import csv
import math
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import xlb.experimental.thermo_mechanical.solid_bounceback as bc
from xlb.utils import save_fields_vtk, save_image
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.multigrid import MultigridSolver


def write_results(norms_over_time, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestep", "l2_disp", "linf_disp", "l2_stress", "linf_stress"])
        writer.writerows(norms_over_time)


if __name__ == "__main__":
    wp.config.mode = "debug"
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP64FP64
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    # initiali1e grid
    nodes_x = 16 * 8
    nodes_y = 16 * 8
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1.0
    length_y = 1.0
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = 10
    dt = 0.01 / 64

    # params
    E = 0.085 * 2.5
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0)
    print("E: {}        nu: {}".format(solid_simulation.E, solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)  # + 3
    manufactured_v = sympy.cos(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x)  # + 3
    expected_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y)

    # get expected stress
    s_xx, s_yy, s_xy = utils.get_expected_stress((manufactured_u, manufactured_v), x, y)
    expected_stress = np.array([
        utils.get_function_on_grid(s_xx, x, y, dx, grid),
        utils.get_function_on_grid(s_yy, x, y, dx, grid),
        utils.get_function_on_grid(s_xy, x, y, dx, grid),
    ])

    # set boundary potential
    potential_sympy = (0.5 - x) ** 2 + (0.5 - y) ** 2 - 0.25
    potential = sympy.lambdify([x, y], potential_sympy)
    indicator = lambda x, y: -1
    boundary_array, boundary_values = bc.init_bc_from_lambda(
        potential_sympy, grid, dx, velocity_set, (manufactured_u, manufactured_v), indicator, x, y
    )

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)
    norms_over_time = list()
    residual_over_time = list()
    multigrid_solver = MultigridSolver(
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        length_x=length_x,
        length_y=length_y,
        dt=dt,
        force_load=force_load,
        gamma=0.8,
        v1=8,
        v2=8,
        max_levels=2,
        boundary_conditions=boundary_array,
        boundary_values=boundary_values,
        potential=potential_sympy,
    )
    finest_level = multigrid_solver.get_finest_level()
    for i in range(timesteps):
        residual_norm = finest_level.start_v_cycle()
        residual_over_time.append(residual_norm)
        macroscopics = finest_level.get_macroscopics()
        l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
        utils.output_image(macroscopics, i, "image1")

        # write out error norms
        # print(finest_level.f_1.numpy()[1,:,:,0])
        # print("-----------------------------------------------------------")

    print(l2_disp, linf_disp, l2_stress, linf_stress)
    print(residual_norm)
    write_results(norms_over_time, "results.csv")
