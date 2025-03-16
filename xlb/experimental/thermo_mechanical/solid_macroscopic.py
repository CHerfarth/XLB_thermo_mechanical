from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
import xlb.experimental.thermo_mechanical.solid_utils as utils


class SolidMacroscopics(Operator):
    def __init__(self, grid, force, boundaries=None, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        self.force = force
        self.boundaries = boundaries
        if self.boundaries == None:
            self.boundaries = grid.create_field(cardinality=10, dtype=precision_policy.store_precision, fill_value=1)

    def _construct_warp(self):
        @wp.kernel
        def kernel(f: Any, displacement: Any, force: Any, boundaries: Any, theta: Any):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            # calculate moments
            f_local = utils.read_local_population(f, i, j)
            m = utils.calc_moments(f_local)

            # apply half-forcing and get displacement
            m[0] += 0.5 * force[0, i, j, 0]
            m[1] += 0.5 * force[1, i, j, 0]
            if boundaries[0, i, j, 0] != wp.int8(0):
                displacement[0, i, j, 0] = m[0]
                displacement[1, i, j, 0] = m[1]
            else:
                displacement[0, i, j, 0] = 0.0
                displacement[1, i, j, 0] = 0.0

            m_eq = utils.calc_equilibrium(m, theta)  # do something with this?

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, displacement, theta):
        wp.launch(
            self.warp_kernel,
            inputs=[f, displacement, self.force, self.boundaries, theta],
            dim=f.shape[1:],
        )
        return displacement.numpy()
