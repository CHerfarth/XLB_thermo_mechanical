import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import jax.numpy as jnp
import numpy as np
import LBMforSolids

#-------------------Setup------------------

# Set up the simulation
domain_size = (100, 100)  # Size of the grid
velocity_field = np.zeros(domain_size)  # Initial velocity field
density_field = np.ones(domain_size)  # Initial density field
tau = 0.6  # Relaxation time

# Instantiate your custom collision operator
custom_collision = LBMforSolids.SolidCollision(omega=1.0)

# Initialize the lattice Boltzmann simulation with the custom collision operator
simulation = xlb.LatticeBoltzmann(domain_size, collision_operator=custom_collision)

# Simulate for some number of steps
for step in range(1000):
    simulation.step(density_field, velocity_field, tau)
    # Perform your other simulation logic here (e.g., boundary conditions, output)