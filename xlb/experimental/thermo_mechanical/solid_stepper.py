from functools import partial
import warp as wp
import numpy as np
import sympy

import xlb
from xlb.operator.stepper import Stepper
from xlb.operator.stream import Stream
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend

from xlb.experimental.thermo_mechanical.solid_collision import SolidsCollision
from xlb.experimental.thermo_mechanical.solid_bounceback import SolidsDirichlet
from xlb.experimental.thermo_mechanical.solid_macroscopic import SolidMacroscopics
import xlb.experimental.thermo_mechanical.solid_utils as utils

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
    def __init__(self, grid, force_load, E, nu, dx, dt, boundary_conditions=None, boundary_values=None, kappa=1, theta=1 / 3):
        super().__init__(grid, boundary_conditions)
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values

        # ----------get material variables------
        mu = E / (2 * (1 + nu))
        lamb = E / (2 * (1 - nu)) - mu
        K = lamb + mu

        # ---------make dimensionless----------
        self.kappa = kappa
        self.L = dx
        self.T = dt
        self.mu = mu * self.T / (self.L * self.L * self.kappa)
        self.lamb = lamb * self.T / (self.L * self.L * self.kappa)
        self.K = K * self.T / (self.L * self.L * self.kappa)
        self.theta = theta

        # ----------calculate omega------------
        omega_11 = 1.0 / (self.mu / theta + 0.5)
        omega_s = 1.0 / (2 * (1 / (1 + theta)) * self.K + 0.5)
        omega_d = 1.0 / (2 * (1 / (1 - theta)) * self.mu + 0.5)
        tau_12 = 0.5
        tau_21 = 0.5
        tau_22 = 0.5
        omega_12 = 1 / (tau_12 + 0.5)
        omega_21 = 1 / (tau_21 + 0.5)
        omega_22 = 1 / (tau_22 + 0.5)
        self.omega = utils.solid_vec(0.0, 0.0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_22, 0.0)

        # ----------handle force load---------
        b_x_scaled = (
            lambda x_node, y_node: force_load[0](x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx) * self.T / self.kappa
        )  # force now dimensionless, and can get called with the indices of the grid nodes
        b_y_scaled = lambda x_node, y_node: force_load[1](x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx) * self.T / self.kappa
        host_force_x = np.fromfunction(
            b_x_scaled, shape=(self.grid.shape[0], self.grid.shape[1])
        )  # create array with force evaluated at the grid points
        print(host_force_x)
        host_force_y = np.fromfunction(b_y_scaled, shape=(self.grid.shape[0], self.grid.shape[1]))
        host_force = np.array([[host_force_x, host_force_y]])
        host_force = np.transpose(host_force, (1, 2, 3, 0))  # swap dims to make array compatible with what grid_factory would have produced
        self.force = wp.from_numpy(host_force, dtype=self.precision_policy.store_precision.wp_dtype)  # ...and move to device

        # ---------define operators----------
        self.collision = SolidsCollision(self.omega, self.force, self.theta)
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.boundaries = SolidsDirichlet()
        self.macroscopic = SolidMacroscopics(
            self.grid, self.force, self.omega, self.theta, self.boundary_conditions, self.velocity_set, self.precision_policy, self.compute_backend
        )
        self.equilibrium = None  # needed?

        #----------create field for temp stuff------------
        self.temp_f = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_current, f_previous):
        wp.launch(utils.copy_populations, inputs=[f_previous, self.temp_f, self.velocity_set.q], dim=f_current.shape[1:])
        wp.launch(self.collision.warp_kernel, inputs=[f_current, self.force, self.omega, self.theta], dim=f_current.shape[1:])
        wp.launch(self.stream.warp_kernel, inputs=[f_current, f_previous], dim=f_current.shape[1:])
        if self.boundary_conditions != None:
            self.boundaries(f_previous, self.temp_f, self.boundary_conditions, self.boundary_values)

    def get_macroscopics(self, f):
        #udate bared moments
        self.macroscopic(f)
        # get updated displacement
        return self.macroscopic.get_macroscopics()
