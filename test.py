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


class Solids2D:
    def __init__(self, omega, prescribed_vel, grid_shape, velocity_set, compute_backend, precision_policy):
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
        self.u = None,
        self.v = None

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def setup_stepper(self):
        #specifications of problem
        E = 0.085*2.5
        nu = 0.8
        mu = E/(2*(1+nu))
        lambda_ =  E/(2*(1-nu)) - mu
        #todo: this needs to be given as input to the solver, not as fixed params!
        L = 1.0
        T = 1
        self.dx = L/self.grid.shape[0]
        dt = 0.01 #arbitrarily set
        kappa = 1
        mu_scaled = (mu*T)/(kappa*L*L)
        b_scaled = np.array([0.1, -0.1]) * (T/(L*kappa))
        lambda_scaled = lambda_ * (T/(L*L*kappa))
        self.stepper = SolidsStepper(
            grid=self.grid,
            force_vector=b_scaled,
            kappa=kappa,
            mu_scaled=mu_scaled,
            lambda_scaled=lambda_scaled
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
                exact_u = 9e-4 * cos(2*pi*(self.dx*i - 0.3)) * sin(2*pi*(self.dx*j + 0.1))
                exact_v = 7e-4 * cos(2*pi*(self.dx*i + 0.4)) * cos(2*pi*(self.dx*j-0.7))
                dif_u = exact_u - self.u[j*self.grid.shape[0] + i]
                dif_v = exact_v - self.v[j*self.grid.shape[0] + i]
                dif = sqrt(dif_u*dif_u + dif_v*dif_v)
                if (L_inf < 0 or dif < L_inf):
                    L_inf = dif
        print(L_inf)


if __name__ == "__main__":
    # Running the simulation
    grid_size = 3
    grid_shape = (grid_size, grid_size)
    compute_backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)

    # Setting fluid viscosity and relaxation parameter.
    Re = 200.0
    prescribed_vel = 0.05
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    simulation = Solids2D(omega, prescribed_vel, grid_shape, velocity_set, compute_backend, precision_policy)
    simulation.run(num_steps=50000000, post_process_interval=1000)