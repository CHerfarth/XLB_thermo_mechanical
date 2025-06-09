import xlb
import time
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
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import argparse


if __name__ == "__main__":
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, compute_backend=compute_backend
    )

    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    parser = argparse.ArgumentParser("LUPs speed test")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("dt", type=float)
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
    timesteps = args.timesteps
    dt = args.dt
    # dt = dx*dx

    # params
    E = 0.085 * 2.5
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    print("E scaled {}, nu {}".format(solid_simulation.E, solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x) * sympy.sin(4 * sympy.pi * x) + 3
    manufactured_v = sympy.cos(2 * sympy.pi * y) * sympy.sin(4 * sympy.pi * x) + 3
    expected_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    print("Mean exp u: {}".format(np.sum(expected_displacement[0, :, :]) * dx * dx))
    print("Mean exp v: {}".format(np.sum(expected_displacement[1, :, :]) * dx * dx))
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
    boundary_array, boundary_values = None, None

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # ------------------------------------- collect data for normal LB ----------------------------------

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    # initialize stepper
    stepper = SolidsStepper(
        grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values
    )

    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    # set initial guess from white noise
    # f_1 = utils.get_initial_guess_from_white_noise(f_2.shape, precision_policy, dx, mean=3, seed=31)

    kernel_provider = KernelProvider()
    copy_populations = kernel_provider.copy_populations
    subtract_populations = kernel_provider.subtract_populations

    wp.synchronize()
    start = time.time()
    for i in range(timesteps):
        stepper(f_1, f_2)
    wp.synchronize()
    end = time.time()

    LUP = f_1.shape[1] * f_1.shape[2] * timesteps
    LUPs = LUP / (end - start)
    MLUPs = LUPs * (1e-6)
    print("MLUP/s: {}".format(MLUPs))
