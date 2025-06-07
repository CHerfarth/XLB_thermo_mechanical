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
from xlb.experimental.thermo_mechanical.multigrid import MultigridSolver
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import argparse
import time


def visualize_smoothing_of_error(
    expected_macroscopics, timesteps, grid, force_load, precision_policy, name="test_", interval=1
):
    gamma = 0.8
    kernel_provider = KernelProvider()
    copy_populations = kernel_provider.copy_populations
    multiply_populations = kernel_provider.multiply_populations
    add_populations = kernel_provider.add_populations
    subtract_populations = kernel_provider.subtract_populations
    l2_norm_squared = kernel_provider.l2_norm
    relaxation_no_defect = kernel_provider.relaxation_no_defect

    stepper = SolidsStepper(grid, force_load, boundary_conditions=None, boundary_values=None)
    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.0

    for i in range(timesteps):
        if (i % interval) == 0:
            macroscopics = stepper.get_macroscopics_host(f_1)
            error_x = expected_macroscopics[0, :, :] - macroscopics[0, :, :, 0]
            if i == 0:
                zmin = np.min(error_x)
                zmax = np.max(error_x)
            utils.plot_3d_surface(error_x, timestep=i, name=name + "_standard", zlim=(zmin, zmax))
        benchmark_data.wu += 1
        stepper(f_1, f_2)
        f_1, f_2 = f_2, f_1

    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    for i in range(timesteps):
        if (i % interval) == 0:
            macroscopics = stepper.get_macroscopics_host(f_1)
            error_x = expected_macroscopics[0, :, :] - macroscopics[0, :, :, 0]
            if i == 0:
                zmin = np.min(error_x)
                zmax = np.max(error_x)
                print(zmin, zmax)
            utils.plot_3d_surface(error_x, timestep=i, name=name + "_relaxed", zlim=(zmin, zmax))
        wp.launch(copy_populations, inputs=[f_1, f_3, 9], dim=f_1.shape[1:])
        stepper(f_1, f_2)
        wp.launch(relaxation_no_defect, inputs=[f_2, f_3, f_1, gamma, 9], dim=f_2.shape[1:])


def write_results(data_over_wu, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "wu",
            "iteration",
            "f_3_norm",
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
    parser.add_argument("max_timesteps_standard", type=int)
    parser.add_argument("E", type=float)
    parser.add_argument("nu", type=float)
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

    # get force load for slow wave
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    manufactured_v = sympy.cos(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x)
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
    visualize_smoothing_of_error(
        expected_macroscopics=expected_macroscopics,
        timesteps=args.max_timesteps_standard,
        grid=grid,
        force_load=force_load,
        precision_policy=precision_policy,
        name="slow",
        interval=3,
    )

    # get force load for fast wave
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(64 * sympy.pi * x) * sympy.sin(64 * sympy.pi * y)
    manufactured_v = sympy.cos(64 * sympy.pi * y) * sympy.sin(64 * sympy.pi * x)
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
    visualize_smoothing_of_error(
        expected_macroscopics=expected_macroscopics,
        timesteps=args.max_timesteps_standard,
        grid=grid,
        force_load=force_load,
        precision_policy=precision_policy,
        name="fast",
        interval=3,
    )
