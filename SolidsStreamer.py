import xlb
from xlb.operator.stream import Stream
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator

import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial


# Mapping:
#    i  j   |   f_q
#    1  0   |   1
#    0  1   |   2
#   -1  0   |   3
#    0 -1   |   4
#    1  1   |   5
#   -1  1   |   6
#   -1 -1   |   7
#    1 -1   |   8
#    0  0   |   9    (irrelevant)

class SolidsStreamerPBC(Stream):
    """
    performs 2d streaming with periodic boundary conditions 
    """
    def __init__(self, dim_x, dim_y, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set, precision_policy, compute_backend)
        self.streamers = list()
        c = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1,1],
            [-1,1],
            [-1,-1],
            [1,-1],
            [0,0],
        ]) 
        for direction in range(9):
            direction_x = c[direction][0]
            direction_y = c[direction][1]
            permutation = list()
            for i in range(dim_x*dim_y):
                old_pos_x = int(i%dim_x)
                old_pos_y = int((i-old_pos_x)/dim_x)
                new_pos_x = (old_pos_x + direction_x)%dim_x
                new_pos_y = (old_pos_y + direction_y)%dim_y
                permutation.append(new_pos_y*dim_x + new_pos_x)
            self.streamers.append(permutation)
        
    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f_0: jnp.ndarray):
        f_post = f_0
        for direction in range(9):
            f_post = f_post.at[direction, :].set(f_0[direction, self.streamers[direction]])
        return f_post