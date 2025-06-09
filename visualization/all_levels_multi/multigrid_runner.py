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
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP64FP64
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, compute_backend=compute_backend
    )

    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    # initiali1e grid
    nodes_x = 128
    nodes_y = 128
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1.0
    length_y = 1.0
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = 10
    dt = dx * dx

    # params
    E = 0.3
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )
    print("E: {}        nu: {}".format(solid_simulation.E, solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.sin(
        2 * sympy.pi * y
    )  # sympy.cos(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    manufactured_v = sympy.sin(
        2 * sympy.pi * y
    )  # sympy.cos(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x)
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
    potential, boundary_array, boundary_values = None, None, None

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
        v1=0,
        v2=1,
        max_levels=None,
        boundary_conditions=boundary_array,
        boundary_values=boundary_values,
        potential=potential_sympy,
        coarsest_level_iter=5000,
    )
    finest_level = multigrid_solver.get_finest_level()
    # finest_level.f_1 = utils.get_initial_guess_from_white_noise(finest_level.f_1.shape, precision_policy, dx, mean=0, seed=39)
    """simulation_params = SimulationParams()
    mu = simulation_params.mu
    K = simulation_params.K
    theta = simulation_params.theta
    x, y = sympy.symbols("x y")
    f = sympy.sin(2*sympy.pi*y)
    df_dx = sympy.diff(f, x)*dt/dx
    df_dy = sympy.diff(f, y)*dt/dx
    mu_11 = -(1+1/(2*mu/theta))*mu*()
    zero = x*0

    f_1_host = utils.get_function_on_grid(f, x, y, dx, grid)
    zero_host = utils.get_function_on_grid(zero, x, y, dx, grid)
    f_1_total = np.array([[f_1_host, f_1_host, f_1_host, f_1_host,f_1_host,f_1_host,f_1_host,f_1_host,f_1_host]])
    host_f_1 = np.transpose(f_1_total, (1, 2, 3, 0))  # swap dims to make array compatible with what grid_factory would have produced
    finest_level.f_1 = wp.from_numpy(host_f_1, dtype=precision_policy.store_precision.wp_dtype)  # ...and move to device"""
    kernel_provider = KernelProvider()
    wp.launch(
        kernel_provider.convert_moments_to_populations,
        inputs=[finest_level.f_1, finest_level.f_1],
        dim=finest_level.f_1.shape[1:],
    )

    for i in range(timesteps):
        residual_norm = finest_level.start_v_cycle(timestep=i, return_residual=True)
        residual_over_time.append(residual_norm)
        macroscopics = finest_level.get_macroscopics()
        l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(
            macroscopics, expected_macroscopics, i, dx, norms_over_time
        )
        # write out error norms
        # print(finest_level.f_1.numpy()[1,:,:,0])
        # print("-----------------------------------------------------------")

    print(l2_disp, linf_disp, l2_stress, linf_stress)
    print(residual_norm)
    write_results(norms_over_time, "results.csv")
