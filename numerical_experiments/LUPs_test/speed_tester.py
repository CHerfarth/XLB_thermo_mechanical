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
    parser = argparse.ArgumentParser("LUPs speed test")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("dt", type=float)
    parser.add_argument("singe_precision", type=int)
    args = parser.parse_args()

    compute_backend = ComputeBackend.WARP
    if args.singe_precision:
        precision_policy = PrecisionPolicy.FP32FP32
    else:
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
    manufactured_u = sympy.cos(2 * sympy.pi * x) * sympy.sin(4 * sympy.pi * x)
    manufactured_v = sympy.cos(2 * sympy.pi * y) * sympy.sin(4 * sympy.pi * x)
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y)

    # get expected stress
    s_xx, s_yy, s_xy = utils.get_expected_stress((manufactured_u, manufactured_v), x, y)
    expected_stress = np.array([
        utils.get_function_on_grid(s_xx, x, y, dx, grid),
        utils.get_function_on_grid(s_yy, x, y, dx, grid),
        utils.get_function_on_grid(s_xy, x, y, dx, grid),
    ])

    
    boundary_array, boundary_values = None, None

    # ------------------------------------- collect data for LB with periodic BC----------------------------------

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

    for i in range(10): #warmup runs to make sure everything compiled
        stepper(f_1, f_2)

    wp.synchronize()
    start = time.time()
    for i in range(timesteps):
        stepper(f_1, f_2)
        f_1, f_2 = f_2, f_1
    wp.synchronize()
    end = time.time()

    LUP = f_1.shape[1] * f_1.shape[2] * timesteps
    LUPs = LUP / (end - start)
    MLUPs = LUPs * (1e-6)
    print("MLUPs Periodic: {}".format(MLUPs))

    # ------------------------------------- collect data for LB with Dirichlet BC----------------------------------

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )

    # set boundary potential
    potential_sympy = (0.5 - x) ** 2 + (0.5 - y) ** 2 - 0.25
    potential = sympy.lambdify([x, y], potential_sympy)
    indicator = lambda x, y: -1
    boundary_array, boundary_values = bc.init_bc_from_lambda(
        potential_sympy, grid, dx, velocity_set, (manufactured_u, manufactured_v), indicator, x, y
    )

    # initialize stepper
    stepper = SolidsStepper(
        grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values
    )

    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    for i in range(10): #warmup runs to make sure everything compiled
        stepper(f_1, f_2, f_3)

    wp.synchronize()
    start = time.time()
    for i in range(timesteps):
        stepper(f_1, f_2, f_3)
        f_1, f_2 = f_2, f_1
    wp.synchronize()
    end = time.time()

    LUP = f_1.shape[1] * f_1.shape[2] * timesteps
    LUPs = LUP / (end - start)
    MLUPs = LUPs * (1e-6)
    print("MLUPs Dirichlet: {}".format(MLUPs))
