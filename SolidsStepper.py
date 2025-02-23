from functools import partial
from jax import jit
import numpy as np

import xlb
from xlb.operator.stepper import Stepper
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend

from SolidsCollision import SolidsCollision

# Mapping:
#    i  j   |   m_q
#    1  0   |   1
#    0  1   |   2
#   -1  0   |   3
#    0 -1   |   4
#    1  1   |   5
#   -1  1   |   6
#   -1 -1   |   7
#    1 -1   |   8
#    0  0   |   9    (irrelevant)

class SolidsStepper(Stepper):
    def __init__(
        self,
        grid,
        force_vector,
        boundary_conditions=[],
        theta=1/3,
        kappa=1, 
        mu_scaled=1,
        lambda_scaled=1,
    ):
        super().__init__(grid, boundary_conditions)

        #compute Omega
        omega_11 = 1 / (mu_scaled / theta + 0.5)
        omega_d = 1 / (2 * mu_scaled / (1 - theta) + 0.5)
        omega_s = 1 / (2 * (mu_scaled + lambda_scaled) / (1 + theta) + 0.5)
        tau_11 = 1 / omega_11 - 0.5
        tau_s = 1 / omega_d - 0.5
        tau_p = 1 / omega_s - 0.5
        tau_12 = 0.5
        tau_21 = tau_12
        tau_22 = 0.5    #ToDo: Check these!!
        omega_12 = 1 / (tau_12 + 0.5)
        omega_21 = 1 / (tau_21 + 0.5)
        omega_22 = 1 / (tau_22 + 0.5)
        omega = np.diag([0, 0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_22, 0])
        
        #calculate force matrix
        self.force = np.zeros((2, self.grid.shape[0]*self.grid.shape[1]))
        self.force[0] = force_vector[0]
        self.force[1] = force_vector[1]
        print(self.force)
        # Construct the collision operator
        self.collision = SolidsCollision(
            velocity_set=self.velocity_set, 
            precision_policy=self.precision_policy, 
            compute_backend=self.compute_backend, 
            theta=theta,
            omega=omega,
            force_vector=self.force)
        
        # Construct Streaming Operator
        print("----------Streaming Operator is ToDo-------------")

    def prepare_fields(self, initializer=None):
        """Prepare the fields required for the stepper.

        Args:
            initializer: Optional operator to initialize the distribution functions.
                        If provided, it should be a callable that takes (grid, velocity_set,
                        precision_policy, compute_backend) as arguments and returns initialized f_0.
                        If None, default equilibrium initialization is used with rho=1 and u=0.

        Returns:
            Tuple of (f_0, f_1, bc_mask, missing_mask):
                - f_0: Initial distribution functions
                - f_1: Copy of f_0 for double-buffering
                - bc_mask: Boundary condition mask indicating which BC applies to each node
                - missing_mask: Mask indicating which populations are missing at boundary nodes
        """
        print("Preparing fields...")
        f_0 = np.zeros((9, self.grid.shape[0] * self.grid.shape[1]))
        f_1 = f_0
        return f_0, f_1, None, None
        
    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, bc_mask, missing_mask):
        """Process boundary conditions and update boundary masks."""
        #ToDo

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_0, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        #ToDo

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f_0, f_1, bc_mask):
        """
        Perform a single step of the lattice boltzmann method
        """
        print("Performing timestep...")
        f_1 = self.collision(f_0)

        return f_0, f_1