import xlb
import sys
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
import xlb.experimental
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import numpy as np
from typing import Any
import sympy
import csv
import math
import xlb.experimental.thermo_mechanical.solid_utils as utils
import argparse
import xlb.experimental.thermo_mechanical.solid_bounceback as bc
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams


def write_results(norms_over_time, name, iteration):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestep", str(iteration) + "_residual"])
        writer.writerows(norms_over_time)


if __name__ == "__main__":
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    # get command line arguments
    parser = argparse.ArgumentParser("error_mode_study")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("dt", type=float)
    parser.add_argument("k", type=int)
    parser.add_argument("output_file", type=str)
    parser.add_argument("iteration", type=int)
    args = parser.parse_args()

    # initialize grid
    nodes_x = args.nodes_x
    nodes_y = args.nodes_y
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1
    length_y = 1
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = args.timesteps
    dt = args.dt

    # get params
    E = 0.085 * 2.5
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_parameters(E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1, theta=1.0 / 3.0)

    # get force load
    k = args.k
    x, y = sympy.symbols("x y")
    manufactured_u = 0  # sympy.sin(2*sympy.pi*x*k)*sympy.sin(2*sympy.pi*y*k)
    # manufactured_u += sympy.sin(2*sympy.pi*x*k)*sympy.cos(sympy.)
    manufactured_v = 0
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
    potential = None
    bc_dirichlet = None
    boundary_array, boundary_values = None, None

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values)

    # startup grids
    f_1 = np.zeros(shape=(9, nodes_x, nodes_y, 1))
    mode = sympy.sin(2 * sympy.pi * k * x) * sympy.sin(2 * sympy.pi * k * y)
    mode += sympy.sin(2 * sympy.pi * k * x) * sympy.cos(2 * sympy.pi * k * y)
    mode += sympy.cos(2 * sympy.pi * k * x) * sympy.sin(2 * sympy.pi * k * y)
    mode += sympy.cos(2 * sympy.pi * k * x) * sympy.cos(2 * sympy.pi * k * y)
    for i in range(9):
        f_1[i, :, :, 0] = utils.get_function_on_grid(mode, x, y, dx, grid)
    f_1 = wp.from_numpy(f_1, dtype=wp.float32)
    # f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    residual_over_time = list()  # to track error over time
    w = 2 / 3

    current = f_1.numpy().copy()
    gamma = 0.8

    l2, linf = 0, 0
    for i in range(timesteps):
        stepper(f_1, f_3)
        current = f_3.numpy().copy() * gamma + (1 - gamma) * current.copy()
        f_1 = wp.from_numpy(current, dtype=wp.float32)
        residual = np.linalg.norm((current).flatten())
        residual_over_time.append((i, residual))

    # write out error norms
    write_results(residual_over_time, args.output_file, args.iteration)
