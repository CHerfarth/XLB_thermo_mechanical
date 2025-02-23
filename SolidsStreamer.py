import xlb
from xlb.operator.stream import Stream
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator

import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial


class SolidsStreamerPBC(Stream):
    """
    performs 2d streaming with periodic boundary conditions 
    """
    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,3,4))
    def jax_implementation(self, f_0: jnp.ndarray, f_1: jnp.ndarray, dim_x: int, dim_y: int):
        #define mapping for c
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
            for i in range(dim_x*dim_y):
                f_1 = f_1.at[direction, (i + direction_y*dim_x + direction_x)%(dim_x*dim_y)].set(f_0[direction, i])
        f_0 = f_1
        return f_0, f_1