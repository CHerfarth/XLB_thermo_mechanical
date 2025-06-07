import jax.numpy as jnp
from jax import jit
import warp as wp
import numpy as np
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet
from functools import partial

import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider


class SolidsCollision(Collision):
    """
    Collision Operator for Solids
    """

    def __init__(
        self,
        omega,
        velocity_set = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        self.omega = omega

    def _construct_warp(self):
        kernel_provider = KernelProvider()
        vec = kernel_provider.vec
        read_local_population = kernel_provider.read_local_population
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global

        @wp.func
        def functional(
            f_vec: vec,
            force_x: self.compute_dtype,
            force_y: self.compute_dtype,
            omega: vec,
            theta: self.compute_dtype,
        ):
            m = calc_moments(f_vec)

            # apply half-forcing and get displacement
            m[0] += self.compute_dtype(0.5) * force_x
            m[1] += self.compute_dtype(0.5) * force_y

            m_eq = calc_equilibrium(m, self.compute_dtype(theta))

            # get post-collision populations
            for l in range(self.velocity_set.q):
                m[l] = omega[l] * m_eq[l] + (self.compute_dtype(1.0) - omega[l]) * m[l]

            # half-forcing
            m[0] += self.compute_dtype(0.5) * force_x
            m[1] += self.compute_dtype(0.5) * force_y

            # get populations
            f_vec_out = calc_populations(m)

            return f_vec_out

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=self.store_dtype),
            f_out: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            f_vec = read_local_population(f, i, j)
            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            f_vec_out = functional(f_vec, force_x, force_y, omega, theta)

            write_population_to_global(f_out, f_vec_out, i, j)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, f_out, force, omega):
        params = SimulationParams()
        theta = params.theta
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f, f_out, force, omega, theta],
            dim=f.shape[1:],
        )
        return fout
