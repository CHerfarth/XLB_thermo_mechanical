from functools import partial
from jax import jit
import numpy as np

import xlb
from xlb.operator.stepper import Stepper
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend

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
        boundary_conditions=[],
        force_vector=None,
        theta=1/3,
    ):
        super().__init__(grid, boundary_conditions)

        #mapping of populations to moments
        self.m_matrix = np.array([
            [ 1,  0, -1,  0,  1, -1, -1,  1],
            [ 0,  1,  0, -1,  1,  1, -1, -1],
            [ 0,  0,  0,  0,  1, -1,  1, -1],
            [ 1,  1,  1,  1,  2,  2,  2,  2],
            [ 1, -1,  1, -1,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  1, -1, -1,  1],
            [ 0,  0,  0,  0,  1,  1, -1, -1],
            [ 0,  0,  0,  0,  1,  1,  1,  1]
        ])

        #mapping of moments to population (inverse of m matrix)
        self.f_matrix = np.array([
            [ 2,  0,  0,  1,  1, -2,  0, -2],
            [ 0,  2,  0,  1, -1,  0, -2, -2],
            [-2,  0,  0,  1,  1,  2,  0, -2],
            [ 0, -2,  0,  1, -1,  0,  2, -2],
            [ 0,  0,  1,  0,  0,  1,  1,  1],
            [ 0,  0, -1,  0,  0, -1,  1,  1],
            [ 0,  0,  1,  0,  0, -1, -1,  1],
            [ 0,  0, -1,  0,  0,  1, -1,  1]
        ]) / 4
        
        self.meq_matrix = np.array([
            [ 1,    0],
            [ 0,    1],
            [ 0,    0],
            [ 0,    0],
            [ 0,    0],
            [theta,  0],
            [ 0, theta],
            [ 0,    0]
        ])

        self.theta = theta

        self.force_vector = force_vector

        # Construct the collision operator
        print("---------ToDo: Construct Collision Operator for SolidsStepper--------------")

        # Construct the operators
        #ToDO
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
        #ToDo
        print("---------Preparing fields is a todo-------------")
        f_0 = [0,0,0,0,0,0,0,0,0]
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
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        print("------SolidsStepper stepping function is a ToDo")
        #manually coding collision, this should be moved to seperate collision operator in the future
        m = self.m_matrix*f_0
        m[0] = m[0] + 0.5 * self.force_vector[0]
        m[1] = m[0] + 0.5 * self.force_vector[1]
        meq = self.meq_matrix*np.array([[m[0]], [m[1]]]) #compute equilibrium moments
        m_post = (1-omega)*m + omega*meq
        m_post[0] = m_post[0] + 0.5*self.force_vector[0]
        m_post[1] = m_post[1] + 0.5*self.force_vector[1]

        return f_0, f_1