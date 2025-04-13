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
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams

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
    def __init__(self, grid, force_load, boundary_conditions=None, boundary_values=None):
        super().__init__(grid, boundary_conditions)
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values

        # get simulation parameters
        params = SimulationParams()
        theta = params.theta
        K = params.K
        mu = params.mu
        dx = params.dx
        T = params.T
        L = params.L
        kappa = params.kappa
        dt = params.dt

        # ----------calculate omega------------
        omega_11 = 1.0 / (mu / theta + 0.5)
        omega_s = 1.0 / (2 * (1 / (1 + theta)) * K + 0.5)
        omega_d = 1.0 / (2 * (1 / (1 - theta)) * mu + 0.5)
        tau_12 = 0.5
        tau_21 = 0.5
        tau_f = 0.5
        omega_12 = 1 / (tau_12 + 0.5)
        omega_21 = 1 / (tau_21 + 0.5)
        omega_f = 1 / (tau_f + 0.5)
        self.omega = utils.solid_vec(0.0, 0.0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_f, 0.0)

        # ----------handle force load---------
        b_x_scaled = lambda x_node, y_node: force_load[0](
            x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx
        )*dt  # force now dimensionless, and can get called with the indices of the grid nodes
        b_y_scaled = lambda x_node, y_node: force_load[1](x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx)*dt
        host_force_x = np.fromfunction(
            b_x_scaled, shape=(self.grid.shape[0], self.grid.shape[1])
        )  # create array with force evaluated at the grid points
        host_force_y = np.fromfunction(b_y_scaled, shape=(self.grid.shape[0], self.grid.shape[1]))
        host_force = np.array([[host_force_x, host_force_y]])
        host_force = np.transpose(host_force, (1, 2, 3, 0))  # swap dims to make array compatible with what grid_factory would have produced
        self.force = wp.from_numpy(host_force, dtype=self.precision_policy.store_precision.wp_dtype)  # ...and move to device
        self.base_force = wp.from_numpy(host_force, dtype=self.precision_policy.store_precision.wp_dtype)

        # ---------define operators----------
        self.collision = SolidsCollision(self.omega)
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.boundaries = SolidsDirichlet(
            boundary_array=self.boundary_conditions,
            boundary_values=self.boundary_values,
            force=self.force,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.macroscopic = SolidMacroscopics(
            self.grid,
            self.force,
            self.omega,
            self.boundary_conditions,
            self.velocity_set,
            self.precision_policy,
            self.compute_backend,
        )
        self.equilibrium = None  # needed?

        # ----------create field for temp stuff------------
        self.temp_f = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_1, f_2):  # f_1 carries current population, f_2 carries the previous post_collision population
        params = SimulationParams()
        theta = params.theta
        wp.launch(utils.copy_populations, inputs=[f_2, self.temp_f, self.velocity_set.q], dim=f_1.shape[1:])
        self.macroscopic(f_1)  # update bared moments (needed for BC)
        # Collision Stage
        wp.launch(self.collision.warp_kernel, inputs=[f_1, self.force, self.omega, theta], dim=f_1.shape[1:])
        # Streaming Stage
        wp.launch(self.stream.warp_kernel, inputs=[f_1, f_2], dim=f_1.shape[1:])
        # Apply BC
        if self.boundary_conditions != None:
            self.boundaries(
                f_destination=f_2,
                f_post_collision=f_1,
                f_previous_post_collision=self.temp_f,
                bared_moments=self.macroscopic.get_bared_moments_device(),
            )
    
    def set_defect_correction(self, defect_correction):
        wp.launch(utils.subtract_populations, inputs=[self.base_force, defect_correction, self.force, 2], dim=self.force.shape[1:])

    def get_macroscopics(self, f):
        # udate bared moments
        self.macroscopic(f)
        # get updated displacement
        return self.macroscopic.get_macroscopics_host()

    def get_macroscopics_device(self, f):
        # udate bared moments
        self.macroscopic(f)
        # get updated displacement
        return self.macroscopic.get_macroscopics_device()