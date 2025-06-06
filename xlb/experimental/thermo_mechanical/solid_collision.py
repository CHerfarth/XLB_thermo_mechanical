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
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)

        self.omega = omega

    def _construct_warp(self):
        # construct warp kernel
        kernel_provider = KernelProvider()
        solid_vec = kernel_provider.solid_vec
        read_local_population = kernel_provider.read_local_population
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global

        @wp.kernel
        def collide(
            f: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: solid_vec,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            # calculate moments
            f_local = read_local_population(f, i, j)
            m = calc_moments(f_local)

            # apply half-forcing and get displacement
            m[0] += self.compute_dtype(0.5) * self.compute_dtype(force[0, i, j, 0])
            m[1] += self.compute_dtype(0.5) * self.compute_dtype(force[1, i, j, 0])

            m_eq = calc_equilibrium(m, self.compute_dtype(theta))

            # get post-collision populations
            for l in range(m._length_):
                m[l] = omega[l] * m_eq[l] + (self.compute_dtype(1.0) - omega[l]) * m[l]

            # half-forcing
            m[0] += self.compute_dtype(0.5) * self.compute_dtype(force[0, i, j, 0])
            m[1] += self.compute_dtype(0.5) * self.compute_dtype(force[1, i, j, 0])

            # get populations and write back to global
            f_local = calc_populations(m)
            write_population_to_global(f, f_local, i, j)

        return None, collide
