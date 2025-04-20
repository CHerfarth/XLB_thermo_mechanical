import numpy as np
import warp as wp
import sympy
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_image
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams


class SolidsDirichlet(Operator):
    """ """

    def __init__(
        self,
        boundary_array,
        boundary_values,
        force,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )
        self.boundary_array = boundary_array
        self.boundary_values = boundary_values
        self.force = force

    def _construct_warp(self):
        opp_indices = self.velocity_set.opp_indices
        w = self.velocity_set.w
        c = self.velocity_set.c_float
        q = self.velocity_set.q

        @wp.func
        def dirichlet_functional(l: Any, i: Any, j: Any, f_current: Any, f_previous: Any, boundary_values: Any, bared_moments: Any, K: Any, mu: Any):
            new_direction = opp_indices[l]
            x_dir = c[0, new_direction]
            y_dir = c[1, new_direction]
            weight = w[new_direction]
            # get values from value array
            u_x = boundary_values[l * 7, i, j, 0]
            u_y = boundary_values[l * 7 + 1, i, j, 0]
            q_ij = boundary_values[l * 7 + 6, i, j, 0]
            # get moments
            m_local = utils.read_local_population(bared_moments, i, j)
            m_s = m_local[3]
            m_d = m_local[4]
            m_11 = m_local[2]
            dx_u_x = -0.25 * (m_s / K + m_d / mu)
            dy_u_y = -0.25 * (m_s / K - m_d / mu)
            cross_dev = -m_11 / mu  # dy_u_x + dx_u_y

            # bounceback with zero order correction
            f_current[new_direction, i, j, 0] = f_previous[l, i, j, 0] + 6.0 * weight * (x_dir * u_x + y_dir * u_y)
            # add first order correction
            if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 1.0) < 1e-3:
                f_current[new_direction, i, j, 0] += 6.0 * weight * (q_ij - 0.5) * (wp.abs(x_dir) * dx_u_x + wp.abs(y_dir) * dy_u_y)
            if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 2.0) < 1e-3:
                f_current[new_direction, i, j, 0] += 6.0 * weight * (q_ij - 0.5) * (dx_u_x + dy_u_y + x_dir * y_dir * (cross_dev))

        @wp.func
        def vn_functional(
            old_direction: Any,
            i: Any,
            j: Any,
            f_current: Any,
            f_previous: Any,
            boundary_values: Any,
            force: Any,
            bared_moments: Any,
            K: wp.float32,
            mu: wp.float32,
            tau_t: wp.float32,
        ):
            theta = 1.0 / 3.0  # assuming this, see issue on github
            new_direction = opp_indices[old_direction]
            x_dir = c[0, new_direction]
            y_dir = c[1, new_direction]
            weight = w[new_direction]
            # read out values
            n_x = boundary_values[old_direction * 7, i, j, 0]
            n_y = boundary_values[old_direction * 7 + 1, i, j, 0]
            T_x = boundary_values[old_direction * 7 + 2, i, j, 0]
            T_y = boundary_values[old_direction * 7 + 3, i, j, 0]
            q_ij = boundary_values[old_direction * 7 + 6, i, j, 0]
            # get zeta
            zeta = 1.0
            if wp.abs(n_x) > wp.abs(n_y):
                zeta = -1.0
            # get c's
            c_1 = -(2.0 * (1.0 - theta) * (K - mu)) / (theta * (1.0 - theta - 4.0 * mu))
            c_2 = -(2.0 * mu) / (theta - 2.0 * mu)
            c_3 = -(4.0 * mu) / (1.0 - theta - 4.0 * mu)

            local_sum = 0.0
            if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 1.0) < 1e-3:  # case V1
                local_sum = 0.0
                for m in range(q):
                    k = c[0, m]
                    l = c[1, m]
                    a_ijkl = wp.abs(k) * wp.abs(l) * (1.0 + x_dir * n_x + y_dir * n_y) * c_1
                    a_ijkl += k * l * (x_dir * n_y + y_dir * n_x) * c_2
                    a_ijkl += (
                        wp.abs(k) * (1.0 - wp.abs(l)) * (wp.abs(x_dir) + x_dir * n_x) + wp.abs(l) * (1.0 - wp.abs(k)) * (wp.abs(y_dir) + y_dir * n_y)
                    ) * c_3
                    if wp.abs(x_dir + k) < 1e-3 and wp.abs(y_dir + l) < 1e-3:
                        a_ijkl += -1.0
                    local_sum += a_ijkl * f_previous[m, i, j, 0]
            elif wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 2.0) < 1e-3:  # case V2
                local_sum = 0.0
                for m in range(q):
                    k = c[0, m]
                    l = c[1, m]
                    a_ijkl = 0.5 * wp.abs(k) * wp.abs(l) * (0.5 * (1.0 + zeta) * x_dir * n_x + 0.5 * (1.0 - zeta) * y_dir * n_y) * c_1
                    a_ijkl += 0.5 * k * l * (x_dir * y_dir + 0.5 * (1.0 + zeta) * x_dir * n_y + 0.5 * (1.0 - zeta) * y_dir * n_x) * c_2
                    a_ijkl += (
                        0.5
                        * (
                            wp.abs(k) * (1.0 - wp.abs(l)) * 0.5 * (1.0 + zeta) * x_dir * n_x
                            + wp.abs(l) * (1.0 - wp.abs(k)) * 0.5 * (1.0 - zeta) * y_dir * n_y
                        )
                        * c_3
                    )
                    if wp.abs(x_dir + k) < 1e-3 and wp.abs(y_dir + l) < 1e-3:
                        a_ijkl += -1.0
                    local_sum += a_ijkl * f_previous[m, i, j, 0]

            f_current[new_direction, i, j, 0] = local_sum
            if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 1.0) < 1e-3:
                f_current[new_direction, i, j, 0] += T_x * x_dir + T_y * y_dir
            elif wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 2.0) < 1e-3:
                f_current[new_direction, i, j, 0] += 0.25 * (x_dir * (1.0 + zeta) * T_x + y_dir * (1.0 - zeta) * T_y)
            # get derivatives of stress
            m_local = utils.read_local_population(bared_moments, i, j)
            m_10 = m_local[0]
            m_01 = m_local[1]
            m_11 = m_local[2]
            m_s = m_local[3]
            m_d = m_local[4]
            m_12 = m_local[5]
            m_21 = m_local[6]
            m_f = m_local[7]

            dx_sxx = 2.0 * (theta * m_10 - m_12) / (1.0 + 2.0 * tau_t) - force[0, i, j, 0]
            dy_syy = 2.0 * (m_12 - theta * m_10) / (1.0 + 2.0 * tau_t) - force[1, i, j, 0]
            dy_sxy = 2.0 * (m_12 - theta * m_10) / (1.0 + 2.0 * tau_t)
            dx_sxy = 2.0 * (m_21 - theta * m_01) / (1.0 + 2.0 * tau_t)

        @wp.kernel
        def bc_kernel(
            f_post_stream: Any,
            f_post_collision: Any,
            f_previous_post_collision: Any,
            boundary_array: Any,
            boundary_values: Any,
            force: Any,
            bared_moments: Any,
            K: Any,
            mu: Any,
        ):
            i, j, k = wp.tid()  # for 2d k will equal 1
            tau_t = 0.5  # ToDo: as argument of fuction
            if boundary_array[0, i, j, 0] == wp.int8(0):  # if outside domain, just set to 0
                for l in range(q):
                    f_post_stream[l, i, j, 0] = wp.nan
            elif boundary_array[0, i, j, 0] == wp.int8(2):  # for boundary nodes: check which directions need to be given by dirichlet BC
                for l in range(q):
                    if boundary_array[l + 1, i, j, 0] == wp.int8(
                        1
                    ):  # this means the interior node is connected to a ghost node in direction l; the bounce back bc needs to be applied
                        dirichlet_functional(l, i, j, f_post_stream, f_previous_post_collision, boundary_values, bared_moments, K, mu)
            elif boundary_array[0, i, j, 0] == wp.int8(3):  # for boundary nodes: check which directions need to be given VN BC
                for l in range(q):
                    if boundary_array[l + 1, i, j, 0] == wp.int8(1):
                        # print("Calling Von Neumann")
                        vn_functional(l, i, j, f_post_stream, f_post_collision, boundary_values, force, bared_moments, K, mu, tau_t)

        return (dirichlet_functional, vn_functional), bc_kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_destination, f_post_collision, f_previous_post_collision, bared_moments):
        # Launch the warp kernel
        params = SimulationParams()
        K = params.K
        mu = params.mu

        wp.launch(
            self.warp_kernel,
            inputs=[
                f_destination,
                f_post_collision,
                f_previous_post_collision,
                self.boundary_array,
                self.boundary_values,
                self.force,
                bared_moments,
                K,
                mu,
            ],
            dim=f_destination.shape[1:],
        )


# --------------utils used to construct bc arrays----------------
def init_bc_from_lambda(potential_sympy, grid, dx, velocity_set, manufactured_displacement, indicator, x, y):
    # Mapping:
    # 0: ghost node
    # 1: interior node
    # 2: boundary node with dirichlet bc
    # 3: boundary node with VN bc

    # Mapping for boundary values, dirichlet:
    # 0: u_x
    # 1: u_y
    # 2:
    # 3:
    # previoui 4:
    # 5:
    # 6: q_ij

    # Mapping for boundary values, VN:
    # 0: n_x
    # 1: n_y
    # 2: T_x
    # 3: T_y
    # 4:
    # 5:
    # 6: q_ij

    potential = sympy.lambdify([x, y], potential_sympy)

    params = SimulationParams()
    T = params.T
    L = params.L
    K = params.K
    mu = params.mu
    kappa = params.kappa

    values_per_direction = 7
    host_boundary_info = np.zeros(shape=(19, grid.shape[0], grid.shape[1], 1), dtype=np.int8)
    host_boundary_values = np.zeros(
        shape=(velocity_set.q * values_per_direction, grid.shape[0], grid.shape[1], 1), dtype=np.float32
    )  # todo: change to compute precision

    # lambdify bc
    bc_dirichlet = [sympy.lambdify([x, y], manufactured_displacement[0]), sympy.lambdify([x, y], manufactured_displacement[1])]

    # get derivative of BC
    bc_dirichlet_devs = [
        sympy.diff(manufactured_displacement[0], x),
        sympy.diff(manufactured_displacement[0], y),
        sympy.diff(manufactured_displacement[1], x),
        sympy.diff(manufactured_displacement[1], y),
    ]
    # lambdify derivatives
    for i in range(4):
        bc_dirichlet_devs[i] = sympy.lambdify([x, y], bc_dirichlet_devs[i])

    # concatenate with non-derivative BC
    bc_dirichlet = np.concatenate((bc_dirichlet, bc_dirichlet_devs))

    # step 1: set all nodes with negative potential to interior
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if potential(i * dx + 0.5 * dx, j * dx + 0.5 * dx) <= 0:
                host_boundary_info[0, i, j, 0] = 1

    # get gradients of potential
    dx_potential = sympy.lambdify([x, y], sympy.diff(potential_sympy, x))
    dy_potential = sympy.lambdify([x, y], sympy.diff(potential_sympy, y))

    # step 2: for each interior node, check if all neighbor nodes are also interior; if not, set to boundary
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if host_boundary_info[0, i, j, 0] == 1:
                for direction in range(velocity_set.q):
                    x_direction = velocity_set._c[0, direction]
                    y_direction = velocity_set._c[1, direction]
                    on_boundary = False
                    on_boundary = (
                        on_boundary or i + x_direction < 0 or i + x_direction >= grid.shape[0]
                    )  # check if on edge of grid, automatically counts as boundary node too
                    on_boundary = on_boundary or j + y_direction < 0 or j + y_direction >= grid.shape[1]

                    # check if boundary node
                    boundary_node = False
                    cur_x, cur_y = i * dx + 0.5 * dx, j * dx + 0.5 * dx
                    bc_x, bc_y = cur_x, cur_y
                    q_ij = 0.5

                    if on_boundary:  # if already on boundary of grid, just move the 0.5 to the side
                        boundary_node = True
                        bc_x = cur_x + 0.5 * dx * x_direction
                        bc_y = cur_y + 0.5 * dx * y_direction
                    elif host_boundary_info[0, (i + x_direction), (j + y_direction), 0] == 0:
                        boundary_node = True
                        max_steps = 100
                        stepsize = dx / max_steps
                        counter = 0
                        while (
                            potential(bc_x, bc_y) < 0
                        ):  # otherwise distance to boundary needs to be found, move along direction of pathway until on boundary
                            bc_x += stepsize * x_direction
                            bc_y += stepsize * y_direction
                            counter += 1
                            assert counter <= (max_steps + 1)
                        q_ij = counter / max_steps

                    # if dirichlet bc: set values
                    if boundary_node and indicator(bc_x, bc_y) < 0:
                        host_boundary_info[direction + 1, i, j, 0] = 1
                        host_boundary_info[0, i, j, 0] = 2
                        for k in range(values_per_direction - 1):
                            host_boundary_values[direction * values_per_direction + k, i, j, 0] = bc_dirichlet[k](bc_x, bc_y)
                        host_boundary_values[(direction + 1) * values_per_direction - 1, i, j, 0] = q_ij
                    elif boundary_node and indicator(bc_x, bc_y) > 0:  # if VN: find T
                        host_boundary_info[direction + 1, i, j, 0] = 1
                        opposite_direction = velocity_set.opp_indices[direction]
                        host_boundary_info[1 + 9 + opposite_direction, i, j, 0] = 1
                        host_boundary_info[0, i, j, 0] = 3
                        # find n
                        n = [dx_potential(bc_x, bc_y), dy_potential(bc_x, bc_y)]
                        n = n / (np.linalg.norm(n))  # normalise
                        # if on edge of domain: normals cant be calculated with potential
                        if on_boundary:
                            n = [0.0, 0.0]
                            if i + x_direction < 0 or i + x_direction >= grid.shape[0]:
                                n[0] = x_direction
                                n[1] = 0
                            if j + y_direction < 0 or j + y_direction >= grid.shape[1]:
                                n[0] = 0
                                n[1] = y_direction

                        # find T
                        dx_ux = bc_dirichlet[2](bc_x, bc_y)
                        dy_ux = bc_dirichlet[3](bc_x, bc_y)
                        dx_uy = bc_dirichlet[4](bc_x, bc_y)
                        dy_uy = bc_dirichlet[5](bc_x, bc_y)
                        T_x = ((K - mu) * (dx_ux + dy_uy) * n[0] + mu * (2 * dx_ux * n[0] + (dx_uy + dy_ux) * n[1])) * L * kappa
                        T_y = ((K - mu) * (dx_ux + dy_uy) * n[1] + mu * ((dx_uy + dy_ux) * n[0] + 2 * dy_uy * n[1])) * L * kappa

                        # write to array
                        host_boundary_values[direction * values_per_direction, i, j, 0] = n[0]
                        host_boundary_values[direction * values_per_direction + 1, i, j, 0] = n[1]
                        host_boundary_values[direction * values_per_direction + 2, i, j, 0] = T_x
                        host_boundary_values[direction * values_per_direction + 3, i, j, 0] = T_y
                        host_boundary_values[(direction + 1) * values_per_direction - 1, i, j, 0] = q_ij

    # move to device
    return wp.from_numpy(host_boundary_info, dtype=wp.int8), wp.from_numpy(host_boundary_values, dtype=wp.float32)
