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
import copy
import xlb.experimental.thermo_mechanical.solid_bounceback as bc


def write_results(norms_over_time, name):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestep', 'l2_disp','linf_disp','l2_stress', 'linf_stress'])
        writer.writerows(norms_over_time)


if __name__ == "__main__":
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    #get command line arguments
    parser = argparse.ArgumentParser("convergence_study")
    parser.add_argument("nodes_x", type=int)
    parser.add_argument("nodes_y", type=int)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("dt", type=float)
    parser.add_argument("include_bc", type=int)
    parser.add_argument("post_process_interval", type=int)
    parser.add_argument("output_file",type=str)
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
    mu = E / (2 * (1 + nu))
    lamb = E / (2 * (1 - nu)) - mu
    K = lamb + mu


    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = 3*sympy.cos(6*sympy.pi*x)*sympy.sin(6*sympy.pi*y)
    manufactured_v = 3*sympy.cos(6*sympy.pi*y)*sympy.sin(6*sympy.pi*x)
    expected_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y, mu, K)

    #get expected stress
    s_xx, s_yy, s_xy = utils.get_expected_stress((manufactured_u, manufactured_v), x, y, lamb, mu)
    expected_stress = np.array([utils.get_function_on_grid(s_xx, x, y, dx, grid), utils.get_function_on_grid(s_yy, x, y, dx, grid), utils.get_function_on_grid(s_xy, x, y, dx, grid)])


    # set boundary potential
    potential = lambda x, y: (0.5-x)**2 + (0.5-y)**2 - 0.25
    boundary_array, boundary_values = bc.init_bc_from_lambda(potential, grid, dx, velocity_set, (manufactured_u, manufactured_v), x, y)
    if args.include_bc == 0:
        potential = None
        bc_dirichlet = None
        boundary_array, boundary_values = None, None


    #adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(grid, force_load, E, nu, dx, dt, boundary_conditions=boundary_array, boundary_values=boundary_values)


    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    norms_over_time = list() #to track error over time
    post_process_interval = args.post_process_interval
    tolerance = 1e-3
    old_macroscopics = list()
    converged = False

    l2, linf = 0, 0
    for i in range(timesteps):
        stepper(f_1, f_3)
        f_1, f_2, f_3 = f_3, f_1, f_2
        if (i % post_process_interval == 0):
            macroscopics = stepper.get_macroscopics(f_1)
            for old_macroscopic in old_macroscopics:
                if np.linalg.norm(old_macroscopic - macroscopics) < tolerance:
                    converged = True
                print(np.linalg.norm(old_macroscopic - macroscopics))
            old_macroscopics.append(macroscopics)
            utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
            if converged: break

    macroscopics = stepper.get_macroscopics(f_1)
    utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
    #write out error norms
    last_norms = norms_over_time[len(norms_over_time)-1]
    print("Final error L2_disp: {}".format(last_norms[1]))
    print("Final error Linf_disp: {}".format(last_norms[2]))
    print("Final error L2_stress: {}".format(last_norms[3]))
    print("Final error Linf_stress: {}".format(last_norms[4]))
    print("Final iteration: {}".format(i))
    #print("in {} timesteps".format(last_norms[0]))
    write_results(norms_over_time, args.output_file)