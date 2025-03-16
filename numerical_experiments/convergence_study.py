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


def output_image(displacement_device, timestep, name):
    #get current displacement ToDo: stress?
    displacement_host = utils.get_macroscopics(displacement_device)
    dis_x = displacement_host[0, :, :, 0]
    dis_y = displacement_host[1, :, :, 0]
    #output as vtk files
    dis_mag = np.sqrt(np.square(dis_x) + np.square(dis_y))
    fields = {"dis_x": dis_x, "dis_y": dis_y, "dis_mag": dis_mag}
    save_fields_vtk(fields, timestep=timestep, prefix=name)


def process_error(displacement_device, manufactured_displacement, timestep, dx, norms_over_time):
    #get current displacement ToDo: stress?
    displacement_host = utils.get_macroscopics(displacement_device)
    #calculate error to expected solution
    l2, linf = utils.get_error_norm(displacement_host[:,:,:,0], manufactured_displacement, dx)
    norms_over_time.append((i, l2, linf))
    return l2, linf

def write_results(norms_over_time, name):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestep', 'L2','Linf'])
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
    parser.add_argument("post_process_interval", type=int)
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
    manufactured_u = 9 * sympy.cos(2*sympy.pi*x)*sympy.sin(2*sympy.pi*y)
    manufactured_v = 7 * sympy.sin(2*sympy.pi*x)*sympy.cos(2*sympy.pi*y)
    manufactured_displacement = np.array([utils.get_function_on_grid(manufactured_u, x, y, dx, grid), utils.get_function_on_grid(manufactured_v, x, y, dx, grid)])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y, mu, K)
    stepper = SolidsStepper(grid, force_load, E, nu, dx, dt)

    # startup grids
    f_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    displacement = grid.create_field(cardinality=2, dtype=precision_policy.store_precision)

    norms_over_time = list() #to track error over time
    post_process_interval = args.post_process_interval
    tolerance = 1e-8

    l2, linf = 0, 0
    for i in range(timesteps):
        stepper(f_0, f_1, displacement)
        f_0, f_1 = f_1, f_0
        if i % post_process_interval == 0:
            l2_new, linf_new = process_error(displacement, manufactured_displacement, i, dx, norms_over_time)
            if math.fabs(l2 - l2_new) < tolerance and math.fabs(linf - linf_new) < tolerance:
                #print("Final timestep:{}".format(i))
                break
            l2, linf = l2_new, linf_new

    #write out error norms
    last_norms = norms_over_time[len(norms_over_time)-1]
    print("Final error L2_disp: {}".format(last_norms[1]))
    print("Final error Linf_disp: {}".format(last_norms[2]))
    print("in {} timesteps".format(last_norms[0]))
    #write_results(norms_over_time, "results.csv")