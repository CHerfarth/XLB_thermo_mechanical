import numpy as np
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_fields_vtk, save_image
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)


class SolidsDirichlet(Operator):
    """ """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    def _construct_warp(self):
        opp_indices = self.velocity_set.opp_indices
        w = self.velocity_set.w
        c = self.velocity_set.c

        @wp.kernel
        def kernel(
            f_pre: Any,
            f_post: Any,
            boundary_array: Any,
            boundary_values: Any,
        ):
            i, j, k = wp.tid()  # for 2d k will equal 1
            if boundary_array[0, i, j, 0] == wp.int8(0):
                for l in range(self.velocity_set.q):
                    f_post[l, i, j, 0] = 0.0
            elif boundary_array[0, i, j, 0] == wp.int8(1):
                for l in range(self.velocity_set.q):
                    if boundary_array[l + 1, i, j, 0] == wp.int8(
                        1
                    ):  # this means the interior node is connected to a ghost node in direction l; the bounce back bc needs to be applied
                        new_direction = opp_indices[l]
                        x_dir = wp.float32(c[new_direction][0])
                        y_dir = wp.float32(c[new_direction][1])  # ToDo: Cast to computational precision
                        weight = w[new_direction]
                        f_post[new_direction, i, j, 0] = f_pre[l, i, j, 0] #+ 6.0 * weight * (x_dir * boundary_values[l*2, i, j, 0] + y_dir * boundary_values[l*2+1, i, j, 0])
                    else:
                        f_post[l, i, j, 0] = f_pre[l, i, j, 0]
            else:
                for l in range(self.velocity_set.q):
                    f_post[l, i, j, 0] = f_pre[l, i, j, 0]

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, boundary_values):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, boundary_values],
            dim=f_pre.shape[1:],
        )
        return f_post


# --------------utils used to construct bc arrays----------------
def init_bc_from_lambda(potential, grid, dx, velocity_set, bc_dirichlet):
    # Mapping:
    # 0: ghost node
    # 1: interior node
    # 2: boundary node
    host_boundary_info = np.zeros(shape=(10, grid.shape[0], grid.shape[1], 1), dtype=np.int8)
    host_boundary_values = np.zeros(shape=(18, grid.shape[0], grid.shape[1], 1), dtype=np.float32) #todo: change to compute precision

    # step 1: set all nodes with negative potential to interior
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if potential(i * dx + 0.5 * dx, j * dx + 0.5 * dx) <= 0:
                host_boundary_info[0, i, j, 0] = 1

    # step 2: for each interior node, check if all neighbor nodes are also interior; if not, set to boundary
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if host_boundary_info[0, i, j, 0] == 1:
                for direction in range(velocity_set.q):
                    x_direction = velocity_set._c[0, direction]
                    y_direction = velocity_set._c[1, direction]
                    on_boundary = False
                    on_boundary = on_boundary or i+x_direction < 0 or i+x_direction>=grid.shape[0] #check if on edge of grid, automatically counts as boundary node too
                    on_boundary = on_boundary or j+y_direction < 0 or j+y_direction>=grid.shape[1]
                    if on_boundary or host_boundary_info[0, (i + x_direction) , (j + y_direction) , 0] == 0:
                        host_boundary_info[direction + 1, i, j, 0] = 1
                        host_boundary_info[0, i, j, 0] = 1
                        #get the boundary condition
                        cur_x, cur_y = i*dx + 0.5*dx, j*dx + 0.5*dx
                        bc_x, bc_y = cur_x, cur_y
                        max_steps = 100
                        stepsize = dx/max_steps
                        counter = 0
                        while (potential(cur_x, cur_y) < 0): #move along direction of pathway until on boundary
                            bc_x += stepsize*x_direction
                            bc_y += stepsize*y_direction
                            counter += 1
                            if counter == max_steps:
                                break
                        if counter == max_steps: #if not able to cross potential in one dx, node must be located on domain boundary. In this case we'll set to the boundary to the condition at the domain boundary
                            bc_x = cur_x + 0.5*dx*x_direction
                            bc_y = cur_y + 0.5*dx*y_direction
                        host_boundary_values[direction*2], host_boundary_values[direction*2 + 1] = bc_dirichlet(bc_x, bc_y)[0], bc_dirichlet(bc_x, bc_y)[1]

    save_image(host_boundary_info[9, :, :, 0], 2)
    # move to device
    return wp.from_numpy(host_boundary_info, dtype=wp.int8), wp.from_numpy(host_boundary_values, dtype=wp.float32)
