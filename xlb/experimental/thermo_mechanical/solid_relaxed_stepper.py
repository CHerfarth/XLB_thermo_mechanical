from functools import partial
import warp as wp
import numpy as np
import sympy
import math

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


class SolidsRelaxedStepper(Stepper):
    def __init__(self, grid, force_load, gamma, boundary_conditions=None, boundary_values=None):
        super().__init__(grid, boundary_conditions)
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values

        self.gamma = gamma

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
        if force_load is not None:
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
            host_force_y = np.fromfunction(
                b_y_scaled, shape=(self.grid.shape[0], self.grid.shape[1])
            )
            host_force = np.array([[host_force_x, host_force_y]])
            host_force = np.transpose(
                host_force, (1, 2, 3, 0)
            )  # swap dims to make array compatible with what grid_factory would have produced
            self.force = wp.from_numpy(
                host_force, dtype=self.precision_policy.store_precision.wp_dtype
            )  # ...and move to device
        else:
            self.force = self.grid.create_field(
                cardinality=2, dtype=self.precision_policy.store_precision
            )

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

        @wp.func
        def functional(
            i: wp.int32,
            j: wp.int32,
            k: wp.int32,
            f_1: wp.array4d(dtype=self.store_dtype),
            defect_vec: vec,
            omega: vec,
            force_x: self.compute_dtype,
            force_y: self.compute_dtype,
            theta: self.compute_dtype,
            gamma: self.compute_dtype,
        ):
            index = wp.vec3i(i, j, k)
            _f_post_collision = read_local_population(f_1, i, j)
            _f_post_stream = self.stream.warp_functional(f_1, index)
            _f_new_post_collision = self.collision.warp_functional(
                f_vec=_f_post_stream, force_x=force_x, force_y=force_y, omega=omega, theta=theta
            )
            _f_out = vec()
            for l in range(self.velocity_set.q):
                _f_out[l] = (
                    gamma * (_f_new_post_collision[l] - defect_vec[l])
                    + (self.compute_dtype(1) - gamma) * _f_post_collision[l]
                )
            return _f_out

        @wp.kernel
        def kernel(
            f_1: wp.array4d(dtype=self.store_dtype),
            f_2: wp.array4d(dtype=self.store_dtype),
            defect_correction: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            theta: self.compute_dtype,
            gamma: self.compute_dtype,
        ):
            i, j, k = wp.tid()
            _defect = read_local_population(defect_correction, i, j)

            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            _f_out = functional(
                i=i,
                j=j,
                k=k,
                f_1=f_1,
                defect_vec=_defect,
                omega=omega,
                force_x=force_x,
                force_y=force_y,
                theta=theta,
                gamma=gamma,
            )

            write_population_to_global(f_2, _f_out, i, j)

        @wp.kernel
        def kernel_residual_norm_squared(
            f_1: wp.array4d(dtype=self.store_dtype),
            res_norm: wp.array1d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()

            _zero_vec = vec()
            for l in range(self.velocity_set.q):
                _zero_vec[l] = self.compute_dtype(0)

            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            _f_pre = read_local_population(f_1, i, j)

            _f_post = functional(
                i=i,
                j=j,
                k=k,
                f_1=f_1,
                defect_vec=_zero_vec,
                omega=omega,
                force_x=force_x,
                force_y=force_y,
                theta=theta,
                gamma=self.compute_dtype(1),
            )

            _local_res = self.compute_dtype(0)
            for l in range(self.velocity_set.q):
                _local_res += (_f_post[l] - _f_pre[l]) * (_f_post[l] - _f_pre[l])

            wp.atomic_add(res_norm, 0, self.store_dtype(_local_res))

        return functional, (kernel, kernel_residual_norm_squared)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self, f_1, f_2, defect_correction
    ):  # f_1 carries current population, f_2 carries the previous post_collision population
        params = SimulationParams()
        theta = params.theta
        if self.boundary_conditions == None:
            wp.launch(
                self.warp_kernel[0],
                inputs=[f_1, f_2, defect_correction, self.force, self.omega, theta, self.gamma],
                dim=f_1.shape[1:],
            )
        else:
            print("Error! Relaxed stepper does not work for boundary conditions yet!")

    def get_residual_norm(self, f):
        params = SimulationParams()
        theta = params.theta
        res_norm = wp.zeros(shape=(1), dtype=self.store_dtype)
        wp.launch(
            self.warp_kernel[1],
            inputs=[f, res_norm, self.force, self.omega, theta],
            dim=f.shape[1:],
        )
        return math.sqrt(1 / (f.shape[0] * f.shape[1] * f.shape[2]) * res_norm.numpy()[0])

    def get_macroscopics(self, f, output_array):
        bared_moments = self.bared_moments(f=f, output_array=output_array, force=self.force)
        return self.macroscopic(
            bared_moments=bared_moments, output_array=bared_moments, force=self.force
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
