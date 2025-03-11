import xlb
import sys
print(sys.path)
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
import math
import configparser
import xlb.experimental.thermo_mechanical.solid_utils as utils

if __name__ == "__main__":

    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, compute_backend=compute_backend
    )

    config = configparser.ConfigParser()
    config.read('simulation_config.ini')
    print(config.sections())

    #initialize grid
    nodes_x = int(config['Grid']['nodes_x'])
    nodes_y = int(config['Grid']['nodes_y'])
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    #get discretization
    length_x= float(config['Domain']['length_x'])
    length_y= float(config['Domain']['length_y'])
    dx = length_x/float(nodes_x)
    dy = length_y/float(nodes_y)
    assert math.isclose(dx, dy)
    total_time = float(config['Time']['total_time'])
    timesteps = int(config['Time']['timesteps'])
    dt = total_time/float(timesteps)

    #get params
    E = float(config['Material Parameters']['E'])
    nu = float(config['Material Parameters']['nu'])
    mu = E / (2 * (1 + nu))
    lamb = E / (2 * (1 - nu)) - mu
    K = lamb + mu

    #get force load
    x, y = sympy.symbols('x y')
    manufactured_u = sympy.sympify(config['Simulation Parameters']['manufactured_u']) 
    manufactured_v = sympy.sympify(config['Simulation Parameters']['manufactured_v']) 
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y, mu, K)
    stepper = SolidsStepper(grid, force_load, E, nu, dx, dt)

    #startup grids
    f_0 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )
    f_1 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )
    displacement = grid.create_field(
        cardinality=2, dtype=precision_policy.store_precision
    )
    for i in range(timesteps):
        stepper(f_0, f_1)


