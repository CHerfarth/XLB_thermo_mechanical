import xlb
from xlb.operator.stream import Stream
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator

import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial


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

class SolidsStreamerPBC(Stream):
    """
    performs 2d streaming with periodic boundary conditions 
    """
    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,2,3))
    def jax_implementation(self, f_0: jnp.ndarray, dim_x: int, dim_y: int):
        print("First call of streamer")
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
        tmp = f_0
        for direction in range(9):
            used = list()
            direction_x = c[direction][0]
            direction_y = c[direction][1]
            for i in range(dim_x*dim_y):
                old_pos_x = int(i%dim_x)
                old_pos_y = int((i-old_pos_x)/dim_x)
                new_pos_x = (old_pos_x + direction_x)%dim_x
                new_pos_y = (old_pos_y + direction_y)%dim_y
                if ((new_pos_y*dim_x + new_pos_x) in used):
                    assert True == False, "Fuck"
                used.append(new_pos_y*dim_x + new_pos_x)
                #print("Old x: {},  Old y: {}, Dir x: {}, Dir y: {}, New x: {} New y: {}".format(old_pos_x, old_pos_y, direction_x, direction_y, new_pos_x, new_pos_y))
                tmp = tmp.at[direction, new_pos_y*dim_x + new_pos_x].set(f_0[direction, i])
            assert len(set(used)) == dim_x*dim_y, "Not nice"
        f_0 = tmp
        print("First call of streamer complete")
        return f_0