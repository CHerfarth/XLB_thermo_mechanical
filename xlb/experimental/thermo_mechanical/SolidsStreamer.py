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
        