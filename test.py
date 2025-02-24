import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np
from SolidsStepper import SolidsStepper
from math import *


class Solids2D: #atm: solves on unit square
    def __init__(self, grid_shape, velocity_set, compute_backend, precision_policy, E, nu, mu, lamb, b_x, b_y, exact_u, exact_v):
        # initialize compute_backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy

        #calculation of dimensionless variables
        self.kappa = 1.
        self.exact_u = exact_u
        self.exact_v = exact_v
        self.U = 1.
        self.L = 3.
        self.T = 1.
        print(type(self.T))
        print(type(mu))
        self.mu_scaled = mu * self.T / (self.L*self.L*self.kappa)
        self.lamb_scaled = lamb * self.T / (self.L*self.L*self.kappa)
        self.dx = self.L/self.grid_shape[0]
        self.b_x_scaled = lambda x, y: b_x(x,y) * self.T / (self.kappa*self.L)
        self.b_y_scaled = lambda x, y: b_y(x,y) * self.T / (self.kappa*self.L)

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def setup_stepper(self):
        self.stepper = SolidsStepper(
            grid=self.grid,
            force_vector= [self.b_x_scaled, self.b_y_scaled],
            kappa=self.kappa,
            mu_scaled=self.mu_scaled,
            lambda_scaled=self.lamb_scaled,
            dx=self.dx
        )


    def run(self, num_steps, post_process_interval=100):
        for i in range(num_steps):
            self.f_0, self.f_1, self.u, self.v = self.stepper(self.f_0, self.f_1, self.bc_mask)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        print("Timestep " + str(i))
        L_inf = -1
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                exact_u = self.exact_u(self.dx*i, self.dx*j) 
                exact_v = self.exact_v(self.dx*i, self.dx*j) 
                dif_u = exact_u - self.u[j*self.grid.shape[0] + i]
                dif_v = exact_v - self.v[j*self.grid.shape[0] + i]
                dif = sqrt(dif_u*dif_u + dif_v*dif_v)
                if (L_inf < 0 or dif < L_inf):
                    L_inf = dif
                print(self.u[j*self.grid.shape[0] + i], end=" ")
            print(" ")
        print(L_inf)
        print("\n\n\n")


if __name__ == "__main__":
    # Running the simulation
    grid_size = 3
    grid_shape = (grid_size, grid_size)
    compute_backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)

    #-----------define variables-------------
    E = 0.085*2.5
    nu = 0.8
    mu = E/(2*(1+nu))
    lamb =  E/(2*(1-nu)) - mu
    K = lamb + mu
    #----------define foce load---------------
    b_x = lambda x, y: (mu-K)*cos(x)
    b_y = lambda x, y: (mu-K)*cos(y)
    #----------define exact solution-----------
    exact_u = lambda x, y: cos(x)
    exact_v = lambda x, y: cos(y)
    #-----------start simulation--------------
    simulation = Solids2D(grid_shape, velocity_set, compute_backend, precision_policy, E, nu, mu, lamb, b_x, b_y, exact_u, exact_v)
    simulation.run(num_steps=500000, post_process_interval=1000)