import warp_implementation as wp_impl
import warp as wp
from xlb.grid import grid_factory
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
import xlb.velocity_set
from typing import Any
import numpy as np


def test_streaming():
    nodes_x, nodes_y = 2,2
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, backend=compute_backend
    )
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    f_1 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )
    f_2 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )

    @wp.kernel
    def set_f(f: wp.array4d(dtype=Any)):
        for i in range(9):
            f[i, 0, 0, 0] = 1.
    wp.launch(set_f, inputs=[f_1], dim=1)
    host_f = f_1.numpy()
    print(host_f[0,:,:,0])
    wp.launch(wp_impl.stream, inputs=[f_1, f_2, nodes_x, nodes_y], dim=f_1.shape[1:])

    host_f = f_2.numpy()

    #check if streaming worked
    for i in range (8):
        expected = np.array([[0,0],[0,0]])
        if i == 0: expected[1,0] = 1
        if i == 1: expected[0,1] = 1
        if i == 2: expected[1,0] = 1
        if i == 3: expected[0,1] = 1
        if i == 4: expected[1,1] = 1
        if i == 5: expected[1,1] = 1
        if i == 6: expected[1,1] = 1
        if i == 7: expected[1,1] = 1
        assert np.array_equal(host_f[i, :, :, 0], expected)

