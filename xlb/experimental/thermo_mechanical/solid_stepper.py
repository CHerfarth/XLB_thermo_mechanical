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
from xlb.experimental.thermo_mechanical.solid_bared_moments import SolidBaredMoments
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider

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
        self.omega = KernelProvider().vec(
            0.0, 0.0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_f, 0.0
        )

        # ----------handle force load---------
        b_x_scaled = (
            lambda x_node, y_node: force_load[0](x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx)
            * dt
            / kappa
        )  # force now dimensionless, and can get called with the indices of the grid nodes
        b_y_scaled = (
            lambda x_node, y_node: force_load[1](x_node * dx + 0.5 * dx, y_node * dx + 0.5 * dx)
            * dt
            / kappa
        )
        host_force_x = np.fromfunction(
            b_x_scaled, shape=(self.grid.shape[0], self.grid.shape[1])
        )  # create array with force evaluated at the grid points
        host_force_y = np.fromfunction(b_y_scaled, shape=(self.grid.shape[0], self.grid.shape[1]))
        host_force = np.array([[host_force_x, host_force_y]])
        host_force = np.transpose(
            host_force, (1, 2, 3, 0)
        )  # swap dims to make array compatible with what grid_factory would have produced
        self.force = wp.from_numpy(
            host_force, dtype=self.precision_policy.store_precision.wp_dtype
        )  # ...and move to device

        # ---------define operators----------
        self.collision = SolidsCollision(self.omega)
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.boundaries = SolidsDirichlet(
            force=self.force,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.macroscopic = SolidMacroscopics(
            self.grid,
            self.omega,
            self.velocity_set,
            self.precision_policy,
            self.compute_backend,
        )
        self.bared_moments = SolidBaredMoments(
            self.grid, self.omega, self.velocity_set, self.precision_policy, self.compute_backend
        )
        self.equilibrium = None  # needed?

        # ----------create field for temp stuff------------
        self.temp_f = grid.create_field(
            cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision
        )

    def _construct_warp(self):
        # get kernels
        kernel_provider = KernelProvider()
        copy_populations = kernel_provider.copy_populations
        read_local_population = kernel_provider.read_local_population
        write_population_to_global = kernel_provider.write_population_to_global
        read_bc_info = kernel_provider.read_bc_info
        read_bc_vals = kernel_provider.read_bc_vals
        vec = kernel_provider.vec
        bc_info_vec = kernel_provider.bc_info_vec
        bc_val_vec = kernel_provider.bc_val_vec
        self.copy_populations = kernel_provider.copy_populations

        @wp.kernel
        def kernel_no_bc(
            f_1: wp.array4d(dtype=self.store_dtype),
            f_2: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f_post_collision = read_local_population(f_1, i, j)
            _f_previous_post_collision = read_local_population(f_1, i, j)
            _f_post_stream = self.stream.warp_functional(f_1, index)

            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            _f_post_collision = self.collision.warp_functional(
                f_vec=_f_post_stream, force_x=force_x, force_y=force_y, omega=omega, theta=theta
            )

            write_population_to_global(f_2, _f_post_collision, i, j)

        @wp.kernel
        def kernel_bc(
            f_1: wp.array4d(dtype=self.store_dtype),
            f_2: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            boundary_info: wp.array4d(dtype=wp.int8),
            boundary_vals: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            K: self.compute_dtype,
            mu: self.compute_dtype,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f_post_collision = read_local_population(f_1, i, j)
            _f_previous_post_collision = read_local_population(f_1, i, j)
            _f_post_stream = self.stream.warp_functional(f_1, index)

            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            _bared_m = self.bared_moments.warp_functional(
                f_vec=_f_post_collision, force_x=force_x, force_y=force_y, omega=omega, theta=theta
            )

            _f_post_stream = self.boundaries.warp_functional(
                f_post_stream_vec=_f_post_stream,
                f_post_collision_vec=_f_post_collision,
                f_previous_post_collision_vec=_f_previous_post_collision,
                i=i,
                j=j,
                boundary_info=boundary_info,
                boundary_vals=boundary_vals,
                force_x=force_x,
                force_y=force_y,
                bared_m_vec=_bared_m,
                K=K,
                mu=mu,
                theta=theta,
            )

            _f_post_collision = self.collision.warp_functional(
                f_vec=_f_post_stream, force_x=force_x, force_y=force_y, omega=omega, theta=theta
            )

            write_population_to_global(f_2, _f_post_collision, i, j)

        return None, (kernel_no_bc, kernel_bc)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self, f_1, f_2
    ):  # f_1 carries current population, f_2 carries the previous post_collision population
        params = SimulationParams()
        theta = params.theta
        if self.boundary_conditions == None:
            wp.launch(
                self.warp_kernel[0],
                inputs=[f_1, f_2, self.force, self.omega, theta],
                dim=f_1.shape[1:],
            )
        else:
            K = params.K
            mu = params.mu
            wp.launch(
                self.warp_kernel[1],
                inputs=[
                    f_1,
                    f_2,
                    self.force,
                    self.boundary_conditions,
                    self.boundary_values,
                    self.omega,
                    K,
                    mu,
                    theta,
                ],
                dim=f_1.shape[1:],
            )

    def get_macroscopics(self, f, output_array):
        self.stream(f, output_array)
        self.bared_moments(f=output_array, output_array=output_array, force=self.force)
        return self.macroscopic(
            bared_moments=output_array, output_array=output_array, force=self.force
        )

    def add_boundary_conditions(self, boundary_conditions, boundary_values):
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values
        self.boundaries = SolidsDirichlet(
            force=self.force,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

    def collide(self, f_1, f_2):
        self.collision(f_1, f_2, self.force, self.omega)
