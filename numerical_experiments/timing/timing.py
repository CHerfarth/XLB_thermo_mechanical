import xlb
import sys
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.velocity_set
import warp as wp
import numpy as np
import sympy
import csv
import math
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.multigrid_solver import MultigridSolver
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import argparse
import time


def write_results(data_over_wu, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "wu",
            "iteration",
            "residual_norm",
            "l2_disp",
            "linf_disp",
            "l2_stress",
            "linf_stress",
        ])
        writer.writerows(data_over_wu)


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

    parser = argparse.ArgumentParser("convergence_study")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("max_timesteps_multi", type=int)
    parser.add_argument("max_timesteps_standard", type=int)
    parser.add_argument("E", type=float)
    parser.add_argument("nu", type=float)
    parser.add_argument("test_multigrid", type=int)
    parser.add_argument("test_standard", type=int)
    parser.add_argument("test_relaxed", type=int)
    args = parser.parse_args()

    # initiali1e grid
    nodes_x = args.nodes_x
    nodes_y = args.nodes_y
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1.0
    length_y = 1.0
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    dt = dx * dx

    # params
    E = args.E
    nu = args.nu

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    print("Simulating with E_scaled {}".format(solid_simulation.E))
    print("Simulating with nu {}".format(solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x) * sympy.sin(4 * sympy.pi * x)
    manufactured_v = sympy.cos(2 * sympy.pi * y) * sympy.sin(4 * sympy.pi * x)
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

    potential, boundary_array, boundary_values = None, None, None

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    tol = 1e-7
    gamma = 0.8

    kernel_provider = KernelProvider()
    copy_populations = kernel_provider.copy_populations
    multiply_populations = kernel_provider.multiply_populations
    add_populations = kernel_provider.add_populations
    subtract_populations = kernel_provider.subtract_populations
    l2_norm_squared = kernel_provider.l2_norm
    relaxation_no_defect = kernel_provider.relaxation_no_defect

    # -------------------------------------- collect data for multigrid----------------------------
    converged = 0
    runtime = 0.0
    i = 0
    timesteps = args.max_timesteps_multi
    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.0
    multigrid_solver = MultigridSolver(
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        length_x=length_x,
        length_y=length_y,
        dt=dt,
        force_load=force_load,
        gamma=gamma,
        v1=3,
        v2=3,
        max_levels=None,
    )
    # ------------set initial guess to white noise------------------------
    # --------------------------------------------------------------------
    wp.synchronize()
    # -------------start timing-------------------------------------------
    if args.test_multigrid:
        start = time.time()
        for i in range(timesteps):
            residual_norm = multigrid_solver.start_v_cycle(return_residual=True)
            print(residual_norm)
            if residual_norm / dt < tol:
                converged = 1
                break
        end = time.time()
        runtime = end - start

    print("Multigrid_Converged: {}".format(converged))
    print("Multigrid_Time: {}".format(runtime))
    print("Multigrid_Iterations: {}".format(i))

    del multigrid_solver

    # ------------------------------------- collect data for normal LB ----------------------------------
    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    timesteps = args.max_timesteps_standard
    converged = 0
    runtime = 0.0
    i = 0
    # initialize stepper
    stepper = SolidsStepper(grid, force_load, boundary_conditions=None, boundary_values=None)
    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    residual = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    if args.test_standard:
        wp.synchronize()
        start = time.time()

        for i in range(timesteps):
            if i % 100 == 0:
                wp.launch(copy_populations, inputs=[f_1, residual, 9], dim=f_1.shape[1:])
            stepper(f_1, f_2)
            f_1, f_2 = f_2, f_1
            if i % 100 == 0:
                wp.launch(
                    subtract_populations, inputs=[f_1, residual, residual, 9], dim=f_1.shape[1:]
                )
                residual_norm_sq = wp.zeros(
                    shape=1, dtype=precision_policy.compute_precision.wp_dtype
                )
                wp.launch(
                    l2_norm_squared, inputs=[residual, residual_norm_sq], dim=residual.shape[1:]
                )
                residual_norm = math.sqrt(
                    (1 / (residual.shape[0] * residual.shape[1] * residual.shape[2]))
                    * residual_norm_sq.numpy()[0]
                )
                print(residual_norm)
                if residual_norm / dt < tol:
                    converged = 1
                    break

        end = time.time()
        runtime = end - start

    print("Standard_Converged: {}".format(converged))
    print("Standard_Time: {}".format(runtime))
    print("Standard_Iterations: {}".format(i))
    norms_over_time = list()
    macroscopics = stepper.get_macroscopics(f=f_1, output_array=f_2).numpy()
    utils.process_error(
        macroscopics=macroscopics,
        expected_macroscopics=expected_macroscopics,
        timestep=0,
        dx=dx,
        norms_over_time=norms_over_time,
    )
    last_norms = norms_over_time[len(norms_over_time) - 1]
    print("Final error L2_disp: {}".format(last_norms[1]))
    print("Final error Linf_disp: {}".format(last_norms[2]))
    print("Final error L2_stress: {}".format(last_norms[3]))
    print("Final error Linf_stress: {}".format(last_norms[4]))
    print("in {} timesteps".format(last_norms[0]))

    del f_1
    del f_2
    del residual

    # ------------------------------------- collect data for relaxed LB ----------------------------------
    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    timesteps = args.max_timesteps_standard
    converged = 0
    runtime = 0.0
    i = 0
    # initialize stepper
    stepper = SolidsStepper(grid, force_load, boundary_conditions=None, boundary_values=None)
    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    residual = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    # set initial guess from white noise
    # f_1 = utils.get_initial_guess_from_white_noise(f_1.shape, precision_policy, dx, mean=0, seed=31)

    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.0

    if args.test_relaxed:
        wp.synchronize()
        start = time.time()

        for i in range(timesteps):
            benchmark_data.wu += 1
            wp.launch(copy_populations, inputs=[f_1, residual, 9], dim=f_1.shape[1:])
            stepper(f_1, f_2)
            wp.launch(
                relaxation_no_defect, inputs=[f_2, residual, f_1, gamma, 9], dim=f_2.shape[1:]
            )
            if i % 100 == 0:
                wp.launch(
                    subtract_populations, inputs=[f_1, residual, residual, 9], dim=f_1.shape[1:]
                )
                residual_norm_sq = wp.zeros(
                    shape=1, dtype=precision_policy.compute_precision.wp_dtype
                )
                wp.launch(
                    l2_norm_squared, inputs=[residual, residual_norm_sq], dim=residual.shape[1:]
                )
                residual_norm = math.sqrt(
                    (1 / (residual.shape[0] * residual.shape[1] * residual.shape[2]))
                    * residual_norm_sq.numpy()[0]
                )
                print(residual_norm)
                if residual_norm / dt < tol:
                    converged = 1
                    break

        end = time.time()
        runtime = end - start

    print("Relaxed_Converged: {}".format(converged))
    print("Relaxed_Time: {}".format(runtime))
    print("Relaxed_Iterations: {}".format(i))

    del f_1
    del f_2
    del residual