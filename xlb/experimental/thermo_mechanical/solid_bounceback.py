import numpy as np
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_image
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


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
        c = self.velocity_set.c_float

        @wp.kernel
        def kernel(
            f_current: Any,
            f_previous: Any,
            boundary_array: Any,
            boundary_values: Any,
        ):
            i, j, k = wp.tid()  # for 2d k will equal 1
            if boundary_array[0, i, j, 0] == wp.int8(0): #if outside domain, just set to 0
                for l in range(self.velocity_set.q):
                    f_current[l, i, j, 0] = 0.0
            elif boundary_array[0, i, j, 0] == wp.int8(2): #for boundary nodes: check which directions need to be given by BC
                for l in range(self.velocity_set.q):
                    if boundary_array[l + 1, i, j, 0] == wp.int8(
                        1
                    ):  # this means the interior node is connected to a ghost node in direction l; the bounce back bc needs to be applied
                        new_direction = opp_indices[l]
                        print("------------------------")
                        print("Node:")
                        print(i)
                        print(j)
                        print("Previous direction:")
                        print(l)
                        print(c[0,l])
                        print(c[1,l])
                        print("New Direction:")
                        print(new_direction)
                        print(c[0, new_direction])
                        print(c[1, new_direction])
                        print("Weight")
                        print(w[new_direction])
                        print(boundary_values[l*2, i, j, 0])
                        print(boundary_values[l*2+1, i, j, 0])
                        x_dir = (c[0, new_direction])
                        y_dir = (c[1, new_direction])  
                        weight = w[new_direction]
                        #f_current[new_direction, i, j, 0] = f_previous[l, i, j, 0] + 6.0 * weight * (x_dir * boundary_values[l*2, i, j, 0] + y_dir * boundary_values[l*2+1, i, j, 0])

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_current, f_previous, bc_mask, boundary_values):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_current, f_previous, bc_mask, boundary_values],
            dim=f_current.shape[1:],
        )


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
                    if on_boundary:
                        host_boundary_info[direction + 1, i, j, 0] = 1
                        host_boundary_info[0, i, j, 0] = 2
                        cur_x, cur_y = i*dx + 0.5*dx, j*dx + 0.5*dx
                        bc_x = cur_x + 0.5*dx*x_direction
                        bc_y = cur_y + 0.5*dx*y_direction
                        host_boundary_values[direction*2], host_boundary_values[direction*2 + 1] = bc_dirichlet(bc_x, bc_y)[0], bc_dirichlet(bc_x, bc_y)[1]
                        print("Node {}, {}".format(i, j))
                        print("at: {}, {}".format(cur_x, cur_y))
                        print("Direction: {}, {}".format(x_direction, y_direction))
                        print("Boundary at: {}, {}".format(bc_x, bc_y))
                        print("bc: {}, {}".format(bc_dirichlet(bc_x, bc_y)[0], bc_dirichlet(bc_x, bc_y)[1]))
                    elif host_boundary_info[0, (i + x_direction) , (j + y_direction) , 0] == 0:
                        host_boundary_info[direction + 1, i, j, 0] = 1
                        host_boundary_info[0, i, j, 0] = 2
                        #get the boundary condition
                        cur_x, cur_y = i*dx + 0.5*dx, j*dx + 0.5*dx
                        bc_x, bc_y = cur_x, cur_y
                        max_steps = 100
                        stepsize = dx/max_steps
                        counter = 0
                        #print("{}, {}, {}, {}, {}, {}".format(i,j,cur_x, cur_y, x_direction, y_direction))
                        while (potential(bc_x, bc_y) < 0): #move along direction of pathway until on boundary
                            bc_x += stepsize*x_direction
                            bc_y += stepsize*y_direction
                            counter += 1
                            assert(counter <= max_steps)
                        host_boundary_values[direction*2], host_boundary_values[direction*2 + 1] = bc_dirichlet(bc_x, bc_y)[0], bc_dirichlet(bc_x, bc_y)[1]

    save_image(host_boundary_info[0, :, :, 0], 2)
    # move to device
    return wp.from_numpy(host_boundary_info, dtype=wp.int8), wp.from_numpy(host_boundary_values, dtype=wp.float32)
