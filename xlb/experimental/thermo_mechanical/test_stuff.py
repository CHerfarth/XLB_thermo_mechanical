import warp_implementation as wp_impl
import warp as wp
from xlb.grid import grid_factory
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
import xlb.velocity_set
from typing import Any
import numpy as np


def test_streaming():
    nodes_x, nodes_y = 2, 2
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    @wp.kernel
    def set_f(f: wp.array4d(dtype=Any)):
        for i in range(9):
            f[i, 0, 0, 0] = 1.0

    wp.launch(set_f, inputs=[f_1], dim=1)
    host_f = f_1.numpy()
    print(host_f[0, :, :, 0])
    wp.launch(wp_impl.stream, inputs=[f_1, f_2, nodes_x, nodes_y], dim=f_1.shape[1:])

    host_f = f_2.numpy()

    # check if streaming worked
    for i in range(8):
        expected = np.array([[0, 0], [0, 0]])
        if i == 0:
            expected[1, 0] = 1
        if i == 1:
            expected[0, 1] = 1
        if i == 2:
            expected[1, 0] = 1
        if i == 3:
            expected[0, 1] = 1
        if i == 4:
            expected[1, 1] = 1
        if i == 5:
            expected[1, 1] = 1
        if i == 6:
            expected[1, 1] = 1
        if i == 7:
            expected[1, 1] = 1
        assert np.array_equal(host_f[i, :, :, 0], expected)


def test_conversion_to_moments():
    nodes_x, nodes_y = 1, 1
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    f = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    @wp.func
    def set_f(f: wp.array4d(dtype=Any)):
        for i in range(9):
            f[i, 0, 0, 0] = float(i)

    moments = wp.from_numpy(
        np.zeros(shape=(9), dtype=float), dtype=precision_policy.store_precision.wp_dtype
    )

    @wp.kernel
    def get_moments(f: wp.array4d(dtype=Any), m: wp.array1d(dtype=Any)):
        set_f(f)
        f_vec = wp_impl.read_local(f, 9, 0, 0)
        m_vec = wp_impl.calc_moments(f_vec)
        for i in range(9):
            m[i] = m_vec[i]

    wp.launch(get_moments, inputs=[f, moments], dim=1)

    host_moments = moments.numpy()

    assert host_moments[0] == -2
    assert host_moments[1] == -6


def test_roundtrip_conversion():
    nodes_x, nodes_y = 1, 1
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    @wp.func
    def set_f(f: wp.array4d(dtype=Any)):
        for i in range(9):
            f[i, 0, 0, 0] = float(i)

    @wp.kernel
    def roundtrip(f: wp.array4d(dtype=Any), f_post: wp.array4d(dtype=Any)):
        set_f(f)
        f_vec = wp_impl.read_local(f, 9, 0, 0)
        m_vec = wp_impl.calc_moments(f_vec)
        f_vec = wp_impl.calc_populations(m_vec)
        wp_impl.write_global(f_post, f_vec, 9, 0, 0)

    wp.launch(roundtrip, inputs=[f_1, f_2], dim=1)
    host_f_1 = f_1.numpy()
    host_f2 = f_2.numpy()
    assert np.array_equal(host_f_1[0:8, :, :, :], host_f2[0:8, :, :, :])
