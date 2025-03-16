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
import xlb.experimental.thermo_mechanical.solid_bounceback as bc




def write_results(norms_over_time, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "L2", "Linf"])
        writer.writerows(norms_over_time)


if __name__ == "__main__":
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

    # initialize grid
    nodes_x = 20
    nodes_y = 20
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1
    length_y = 1
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = 5000
    dt = 0.001

    # get params
    E = 0.085 * 2.5
    nu = 0.8
    mu = E / (2 * (1 + nu))
    lamb = E / (2 * (1 - nu)) - mu
    K = lamb + mu


    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = sympy.cos(x*2*sympy.pi)*sympy.cos(y*2*sympy.pi) + 3
    manufactured_v = sympy.cos(y*2*sympy.pi)*sympy.cos(x*2*sympy.pi) + 3
    manufactured_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y, mu, K)
    manufactured_u = sympy.lambdify([x, y], manufactured_u)
    manufactured_v = sympy.lambdify([x, y], manufactured_v)


    # set boundary potential
    potential = lambda x, y: (0.5-x)**2 + (0.5-y)**2 - 0.2 
    bc_dirichlet = lambda x, y: (manufactured_u(x,y), manufactured_v(x,y))
    boundary_array, boundary_values = bc.init_bc_from_lambda(potential, grid, dx, velocity_set, bc_dirichlet)

    #adjust expected solution
    #expected_solution = manufactured_displacement
    expected_solution = utils.restrict_solution_to_domain(manufactured_displacement, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(grid, force_load, E, nu, dx, dt, boundary_conditions=boundary_array, boundary_values=boundary_values)

    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    norms_over_time = list()  # to track error over time
    tolerance = 1e-6

    l2, linf = 0, 0
    for i in range(timesteps):
        stepper(f_1, f_3)
        f_1, f_2, f_3 = f_3, f_1, f_2
        if i % 10 == 0:
            displacement = stepper.get_macroscopics(f_1)
            l2_new, linf_new = utils.process_error(displacement, expected_solution, i, dx, norms_over_time)
            print(l2_new)
            if math.fabs(l2 - l2_new) < tolerance and math.fabs(linf - linf_new) < tolerance:
                print("Final timestep:{}".format(i))
                break
            l2, linf = l2_new, linf_new
            utils.output_image(displacement, i, "figure")

    # write out error norms
    print("Final error: {}".format(norms_over_time[len(norms_over_time) - 1]))
