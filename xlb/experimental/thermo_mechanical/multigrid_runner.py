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
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    # initialize grid
    nodes_x = 80
    nodes_y = 80
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    nodes_x_2 = (int)(nodes_x * 0.5)
    nodes_y_2 = (int)(nodes_y * 0.5)
    grid_2 = grid_factory((nodes_x_2, nodes_y_2), compute_backend=compute_backend)
    print(type(grid))
    print(type(grid_2))

    # get discretization
    length_x = 1
    length_y = 1
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    dx_2 = length_x / float(nodes_x_2)
    timesteps = 100
    dt = 0.001

    # params
    E = 0.085 * 2.5
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_parameters(E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1, theta=1.0 / 3.0)

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x)*sympy.sin(2*sympy.pi*y)  # + 3
    manufactured_v = sympy.cos(2 * sympy.pi * y)*sympy.sin(2*sympy.pi*x)  # + 3
    expected_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y)
    expected_displacement_2 = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx_2, grid_2),
        utils.get_function_on_grid(manufactured_v, x, y, dx_2, grid_2),
    ])

    # get expected stress
    s_xx, s_yy, s_xy = utils.get_expected_stress((manufactured_u, manufactured_v), x, y)
    expected_stress = np.array([
        utils.get_function_on_grid(s_xx, x, y, dx, grid),
        utils.get_function_on_grid(s_yy, x, y, dx, grid),
        utils.get_function_on_grid(s_xy, x, y, dx, grid),
    ])
    expected_stress_2 = np.array([
        utils.get_function_on_grid(s_xx, x, y, dx_2, grid_2),
        utils.get_function_on_grid(s_yy, x, y, dx_2, grid_2),
        utils.get_function_on_grid(s_xy, x, y, dx_2, grid_2),
    ])

    potential, boundary_array, boundary_values = None, None, None

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics_2 = np.concatenate((expected_displacement_2, expected_stress_2), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx_2)
    norms_over_time = list()

    for i in range(timesteps):
        multigrid_solver = MultigridSolver(
            nodes_x=nodes_x,
            nodes_y=nodes_y,
            length_x=length_x,
            length_y=length_y,
            dt=dt,
            E=E,
            nu=nu,
            force_load=force_load,
            gamma=0.8,
            timesteps=i,
            max_levels=2,
        )
        macroscopics = multigrid_solver.work()
        try:
            l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
            utils.output_image(macroscopics, i, "figure1", None, None)
        except:
            l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics_2, i, dx*2, norms_over_time)
            utils.output_image(macroscopics, i, "figure2", None, None)


        # write out error norms
        print(l2_disp, linf_disp, l2_stress, linf_stress)

    write_results(norms_over_time, "results.csv")
    
