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
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
import argparse


def write_results(data_over_wu, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["wu", "iteration", "residual_norm", "l2_disp", "linf_disp", "l2_stress", "linf_stress"])
        writer.writerows(data_over_wu)




if __name__ == "__main__":
    wp.config.mode = "debug"
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
    

    parser = argparse.ArgumentParser("convergence_study")
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
    length_x = 1.
    length_y = 1.
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = args.timesteps
    dt = args.dt

    # params
    E = 0.085 * 2.5
    nu = 0.8

    solid_simulation = SimulationParams()
    solid_simulation.set_parameters(E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1.0, theta=1.0 / 3.0)

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(2 * sympy.pi * x)*sympy.sin(2*sympy.pi*y)  # + 3
    manufactured_v = sympy.cos(2 * sympy.pi * y)*sympy.sin(2*sympy.pi*x)  # + 3
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


    #-------------------------------------- collect data for multigrid----------------------------
    data_over_wu = list()
    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.
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
            v1=2,
            v2=2,
            max_levels=2, #change!!
        )
    finest_level = multigrid_solver.get_finest_level()
    for i in range(timesteps):
        residual_norm = finest_level.start_v_cycle()
        macroscopics = finest_level.get_macroscopics()
        l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics, i, dx, list())
        data_over_wu.append((benchmark_data.wu, i, residual_norm, l2_disp, linf_disp, l2_stress, linf_stress))

    print(l2_disp, linf_disp, l2_stress, linf_stress)
    print(residual_norm)
    write_results(data_over_wu, "multigrid_results.csv")


    #------------------------------------- collect data for normal LB ----------------------------------
    
    # initialize stepper
    stepper = SolidsStepper(grid, force_load, boundary_conditions=None, boundary_values=None)

    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    residual = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    data_over_wu = list()  # to track error over time
    benchmark_data = BenchmarkData()
    benchmark_data.wu = 0.

    l2, linf = 0, 0
    for i in range(timesteps):
        benchmark_data.wu += 1
        wp.launch(utils.copy_populations, inputs=[f_1, residual, 9], dim=f_1.shape[1:])
        stepper(f_1, f_3)
        f_1, f_2, f_3 = f_3, f_1, f_2
        wp.launch(utils.subtract_populations, inputs=[f_1, residual, residual, 9], dim=f_3.shape[1:])
        residual_norm = np.linalg.norm(residual.numpy())

        macroscopics = stepper.get_macroscopics_host(f_1)
        l2_disp, linf_disp, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics, i, dx, list())
        data_over_wu.append((benchmark_data.wu, i, residual_norm, l2_disp, linf_disp, l2_stress, linf_stress))

    print(l2_disp, linf_disp, l2_stress, linf_stress)
    print(residual_norm)
    write_results(data_over_wu, "normal_results.csv")

    
