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
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
        theta=1/3,
        omega=None,
        force_vector=None,
    ):
        
        super().__init__(
            omega=0.6, #NOTE: omega has been removed as an argument in recent vesions of xlb (but not in our version); we set it to an arbitrary value
            velocity_set=velocity_set, 
            precision_policy=precision_policy, 
            compute_backend=compute_backend
        )

        self.omega = omega
        

        #mapping of populations to moments
        self.m_matrix = jnp.array([
            [ 1,  0, -1,  0,  1, -1, -1,  1,0],
            [ 0,  1,  0, -1,  1,  1, -1, -1,0],
            [ 0,  0,  0,  0,  1, -1,  1, -1,0],
            [ 1,  1,  1,  1,  2,  2,  2,  2,0],
            [ 1, -1,  1, -1,  0,  0,  0,  0,0],
            [ 0,  0,  0,  0,  1, -1, -1,  1,0],
            [ 0,  0,  0,  0,  1,  1, -1, -1,0],
            [ 0,  0,  0,  0,  1,  1,  1,  1,0],
            [0,0,0,0,0,0,0,0,0]
        ])

        #mapping of moments to population (inverse of m matrix)
        self.f_matrix = jnp.array([
            [ 2,  0,  0,  1,  1, -2,  0, -2,0],
            [ 0,  2,  0,  1, -1,  0, -2, -2,0],
            [-2,  0,  0,  1,  1,  2,  0, -2,0],
            [ 0, -2,  0,  1, -1,  0,  2, -2,0],
            [ 0,  0,  1,  0,  0,  1,  1,  1,0],
            [ 0,  0, -1,  0,  0, -1,  1,  1,0],
            [ 0,  0,  1,  0,  0, -1, -1,  1,0],
            [ 0,  0, -1,  0,  0,  1, -1,  1,0],
            [0,0,0,0,0,0,0,0,0]
        ]) / 4
        
        self.meq_matrix = jnp.array([
            [ 1,    0],
            [ 0,    1],
            [ 0,    0],
            [ 0,    0],
            [ 0,    0],
            [theta,  0],
            [ 0, theta],
            [ 0,    0],
            [0,0],
        ])

        self.theta = theta

        self.force_vector = force_vector

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray):
        m = jnp.matmul(self.m_matrix,f)
        m = m.at[0,:].add(0.5*self.force_vector[0,:])
        m = m.at[1,:].add(0.5*self.force_vector[1,:])
        u= m[0,:]
        v= m[1,:]
        m_eq = jnp.matmul(self.meq_matrix,m[0:2,:]) 
        m_post = jnp.matmul(self.omega,m_eq) + jnp.matmul((np.eye(9) - self.omega),m)
        m_post = m_post.at[0,:].add(0.5*self.force_vector[0,:])
        m_post = m_post.at[1,:].add(0.5*self.force_vector[1,:])
        f_post = jnp.matmul(self.f_matrix, m_post)
        return f_post, u, v