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


# Mapping for moments:
#    i  j   |   m_q
#    1  0   |   1
#    0  1   |   2
#    1  1   |   3
#    s      |   4
#    d      |   5
#    1  2   |   6
#    2  1   |   7
#    2  2   |   8
#    0  0   |   9 (irrelevant)


class SolidsCollision(Collision):
    """
    Collision Operator for Solids
    """
    def __init__(
        self,
        omega,
        force_matrix,
        theta=1/3,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        
        super().__init__(
            velocity_set=velocity_set, 
            precision_policy=precision_policy, 
            compute_backend=compute_backend
        )

        self.omega = omega

    def _construct_warp(self):

        #construct warp kernel
        @wp.kernel
        def collide(
            f: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            force: wp.array4d(dtype=Any),
            omega: utils.solid_vec,
            theta: Any,
            displacement: wp.array4d(dtype=Any) = None,
        ):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            # calculate moments
            f_local = utils.read_local_population(f, i, j)
            m = utils.calc_moments(f_local)

            # apply half-forcing and get displacement
            m[0] += 0.5 * force[0, i, j, 0]
            m[1] += 0.5 * force[1, i, j, 0]
            if displacement != None:
                displacement[0, i, j, 0] = m[0]
                displacement[1, i, j, 0] = m[1]

            m_eq = utils.calc_equilibrium(m, theta)

            # get post-collision populations
            for l in range(m._length_):  
                m[l] = omega[l] * m_eq[l] + (1.0 - omega[l]) * m[l]

            assert m_eq[0] == m[0]
            assert m_eq[1] == m[1] #sanity check

            # half-forcing
            m[0] += 0.5 * force[0, i, j, 0]
            m[1] += 0.5 * force[1, i, j, 0]

            # get populations and write back to global
            f_local = utils.calc_populations(m)
            utils.write_population_to_globalwrite_global(f_post, f_local, i, j)

        return None, collide
        

        