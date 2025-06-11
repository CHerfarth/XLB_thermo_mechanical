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
from xlb.experimental.thermo_mechanical.multigrid_solver import MultigridSolver
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData


def write_results(norms_over_time, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "WU",
            "Timestep",
            "res norm",
            "L2_disp",
            "Linf_disp",
            "L2_stress",
            "LInf_stress",
        ])
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

    # get command line arguments
    parser = argparse.ArgumentParser("convergence_study")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("include_bc", type=int)
    parser.add_argument("bc_indicator", type=int)
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
    dt = dx*dx 

    # get params
    nu = 0.5
    E = 0.5

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0
    )
    print("E_scaled {}, nu {}".format(solid_simulation.E, solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = 3 * sympy.sin(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    manufactured_v = 3 * sympy.sin(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x)
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
    indicator = lambda x, y: 1 * args.bc_indicator
    boundary_array, boundary_values = bc.init_bc_from_lambda(
        potential_sympy, grid, dx, velocity_set, (manufactured_u, manufactured_v), indicator, x, y
    )
    if args.include_bc == 0:
        potential = None
        bc_dirichlet = None
        boundary_array, boundary_values = None, None

    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)
    macroscopics = grid.create_field(cardinality=9, dtype=precision_policy.store_precision)

    # -------------------------------------- collect data for multigrid----------------------------
    print("Starting simulation...")
    data_over_wu = list()
    residuals = list()
    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.0
    multigrid_solver = MultigridSolver(
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        length_x=length_x,
        length_y=length_y,
        dt=dt,
        force_load=force_load,
        gamma=0.8,
        v1=3,
        v2=3,
        max_levels=1,
        coarsest_level_iter=100,
        boundary_conditions=boundary_array,
        boundary_values=boundary_values,
        potential=potential_sympy,
    )

    for i in range(timesteps):
        residual_norm = multigrid_solver.start_v_cycle(return_residual=True)
        residuals.append(residual_norm)
        multigrid_solver.get_macroscopics(output_array=macroscopics)
        l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(
            macroscopics.numpy(), expected_macroscopics, i, dx, list()
        )
        data_over_wu.append((
            benchmark_data.wu,
            i,
            residual_norm,
            l2_disp,
            linf_disp,
            l2_stress,
            linf_stress,
        ))

    print("Final error L2_disp: {}".format(l2_disp))
    print("Final error Linf_disp: {}".format(linf_disp))
    print("Final error L2_stress: {}".format(l2_stress))
    print("Final error Linf_stress: {}".format(linf_stress))
    print("Residual norm: {}".format(residual_norm))
    #write_results(data_over_wu, "nodes_{}_results.csv".format(nodes_x))
