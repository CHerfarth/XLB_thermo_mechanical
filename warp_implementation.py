from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import numpy as np
from typing import Any


# Mapping (for populations):
#    i  j   |   q
#    1  0   |   0
#    0  1   |   1
#   -1  0   |   2
#    0 -1   |   3
#    1  1   |   4
#   -1  1   |   5
#   -1 -1   |   6
#    1 -1   |   7
#    0  0   |   8 (ignored)



# Mapping (for moments):
#    i  j   |   q
#    1  0   |   0
#    0  1   |   1
#    1  1   |   2
#    s      |   3
#    d      |   4
#    1  2   |   5
#    2  1   |   6
#    2  2   |   7
#    0  0   |   8 (ignored)


precision_policy = PrecisionPolicy.FP32FP32
f_vec = wp.vec(9, dtype=precision_policy.compute_precision.wp_dtype)


@wp.func
def calc_moments(f: f_vec):
    m = f_vec()
    # Todo: find better way to do this!
    m[0] = f[0] - f[2] + f[4] - f[5] - f[6] + f[7]
    m[1] = f[1] - f[3] + f[4] + f[5] - f[6] - f[7]
    m[2] = f[4] - f[5] + f[6] - f[7]
    m[3] = f[0] + f[1] + f[2] + f[3] + 2.0 * f[4] + 2.0 * f[5] + 2.0 * f[6] + 2.0 * f[7]
    m[4] = f[0] - f[1] + f[2] - f[3]
    m[5] = f[4] - f[5] - f[6] + f[7]
    m[6] = f[4] + f[5] - f[6] - f[7]
    m[7] = f[4] + f[5] + f[6] + f[7]
    m[8] = 0.0
    return m


@wp.func
def calc_populations(m: f_vec):
    f = f_vec()
    # Todo: find better way to do this!
    f[0] = 2.0 * m[0] + m[3] + m[4] - 2.0 * m[5] - 2.0 * m[7]
    f[1] = 2.0 * m[1] + m[3] - m[4] - 2.0 * m[6] - 2.0 * m[7]
    f[2] = -2.0 * m[0] + m[3] + m[4] + 2.0 * m[5] - 2.0 * m[7]
    f[3] = -2.0 * m[1] + m[3] - m[4] + 2.0 * m[6] - 2.0 * m[7]
    f[4] = m[2] + m[5] + m[6] + m[7]
    f[5] = -m[2] - m[5] + m[6] + m[7]
    f[6] = m[2] - m[5] - m[6] + m[7]
    f[7] = -m[2] + m[5] - m[6] + m[7]
    f[8] = 0.0
    for i in range(9):
        f[i] = f[i]/4.
    return f


@wp.func
def read_local(f: wp.array4d(dtype=Any), dim: wp.int32, x: wp.int32, y: wp.int32):
    f_local = f_vec()
    for i in range(dim):
        f_local[i] = f[i, x, y, 0]
    return f_local


@wp.func
def write_global(
    f: wp.array4d(dtype=Any), f_local: f_vec, dim: wp.int32, x: wp.int32, y: wp.int32
):
    for i in range(dim):
        f[i, x, y, 0] = f_local[i]


@wp.func
def calc_equilibrium(m: f_vec, theta: Any):
    m_eq = f_vec()
    m_eq[0] = m[0]
    m_eq[1] = m[1]
    m_eq[2] = 0.0
    m_eq[3] = 0.0
    m_eq[4] = 0.0
    m_eq[5] = theta * m[0]
    m_eq[6] = theta * m[1]
    m_eq[7] = 0.0
    m_eq[8] = 0.0
    return m_eq


@wp.kernel
def collide(
    f: wp.array4d(dtype=Any),
    f_post: wp.array4d(dtype=Any),
    force: wp.array4d(dtype=Any),
    displacement: wp.array4d(dtype=Any),
    omega: f_vec,
    theta: Any,
):
    i, j, k = wp.tid()  # for 2d, k will equal 1

    # calculate moments
    f_local = read_local(f, 9, i, j)
    m = calc_moments(f_local)

    # apply half-forcing and get displacement
    m[0] += 0.5 * force[0, i, j, 0]
    m[1] += 0.5 * force[1, i, j, 0]
    displacement[0, i, j, 0] = m[0]
    displacement[1, i, j, 0] = m[1]

    m_eq = calc_equilibrium(m, theta)

    # get post-collision populations
    for l in range(m._length_):
        m[l] = omega[l] * m_eq[l] + (1.0 - omega[l]) * m[l]

    # half-forcing
    m[0] += 0.5 * force[0, i, j, 0]
    m[1] += 0.5 * force[1, i, j, 0]

    # get populations and write back to global
    f_local = calc_populations(m)
    write_global(f_post, f_local, 9, i, j)


@wp.kernel
def stream(
    f: wp.array4d(dtype=Any),
    f_post: wp.array4d(dtype=Any),
    dim_x: wp.int32,
    dim_y: wp.int32,
):
    i, j, k = wp.tid()

    for direction in range(9):
        dir_x, dir_y = 0, 0
        # get directions to stream in (Todo: make more elegant/reusable/modular)
        if direction == 0:
            dir_x, dir_y = 1, 0
        elif direction == 1:
            dir_x, dir_y = 0, 1
        elif direction == 2:
            dir_x, dir_y = -1, 0
        elif direction == 3:
            dir_x, dir_y = 0, -1
        elif direction == 4:
            dir_x, dir_y = 1, 1
        elif direction == 5:
            dir_x, dir_y = -1, 1
        elif direction == 6:
            dir_x, dir_y = -1, -1
        elif direction == 7:
            dir_x, dir_y = 1, -1

        new_i = wp.mod((i + dir_x + dim_x) , dim_x)
        new_j = wp.mod((j + dir_y + dim_y), dim_y)

        f_post[direction, new_i, new_j, k] = f[direction, i, j, k]
        #if direction == 2: wp.printf("Previous: i=%d, j=%d, val=%f         now at: i=%d, j=%d\n", i, j, f[direction, i, j, k], new_i, new_j)

def post_process(displacement: wp.array4d(dtype=Any), L, T, kappa):
    displacement_host = displacement.numpy()
    #perform rescaling
    displacement_host = displacement_host * L *kappa / T
    print(
        np.linalg.norm(displacement_host[0, :, :, 0]),
        np.linalg.norm(displacement_host[1, :, :, 0]),
    )
    #print(displacement_host[0,:,:,0])


if __name__ == "__main__":
    # set dimensions of domain
    domain_x = 3  # for now we work on square
    domain_y = 3

    # total time
    total_time = 20

    # set shape of grid
    nodes_x = 30
    nodes_y = 30
    timesteps = 400

    # calculate dx, dt
    dx = domain_x / (nodes_x - 1)
    dy = domain_y / (nodes_y - 1)
    print(dx, dy)
    dt = total_time / timesteps
    assert dx == dy, "Wrong spacial steps in directions x and y do not match"

    # init xlb stuff
    compute_backend = ComputeBackend.WARP
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, backend=compute_backend
    )

    # initialise grid (should probably move this to Stepper class for full implementation later)
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    # vector_size = 3 #f_pre, f_eq, f_post
    f_1 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )
    f_2 = grid.create_field(
        cardinality=velocity_set.q, dtype=precision_policy.store_precision
    )
    displacement = grid.create_field(
        cardinality=2, dtype=precision_policy.store_precision
    )

    # print startup info
    print("Initialized grid with dimensions     {}x{}".format(nodes_x, nodes_y))

    # -----------define variables-------------
    E = 0.085 * 2.5
    nu = 0.8
    mu = E / (2 * (1 + nu))
    lamb = E / (2 * (1 - nu)) - mu
    K = lamb + mu
    theta = 1 / 3

    # -----------make dimensionless----------
    L = dx
    T = dt
    kappa = 1
    mu_scaled = mu * T / (L * L * kappa)
    lamb_scaled = lamb * T / (L * L * kappa)
    K_scaled = K * T / (L * L * kappa)

    # calculate omega
    omega_11 = 1 / (mu_scaled / theta + 0.5)
    omega_s = 1 / (2 * (1 / (1 + theta)) * K_scaled + 0.5)
    omega_d = 1 / (2 * (1 / (1 - theta)) * mu_scaled + 0.5)
    tau_12 = 0.5
    tau_21 = 0.5
    tau_22 = 0.5
    omega_12 = 1 / (tau_12 + 0.5)
    omega_21 = 1 / (tau_21 + 0.5)
    omega_22 = 1 / (tau_22 + 0.5)
    omega = f_vec(
        0.0, 0.0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_22, 0.0
    )

    # ----------define foce load---------------
    b_x = lambda x, y: (mu - K) * (np.cos(x))
    b_y = lambda x, y: (mu - K) * (np.cos(y))
    # make dimensionless and in terms of node positions
    b_x_scaled = lambda i, j: b_x(i * dx, j * dx) * T / kappa
    b_y_scaled = lambda i, j: b_y(i * dx, i * dx) * T / kappa
    host_force_x = np.fromfunction(b_x_scaled, shape=(nodes_x, nodes_y))
    host_force_y = np.fromfunction(b_y_scaled, shape=(nodes_x, nodes_y))
    host_force = np.array([[host_force_x, host_force_y]])
    host_force = np.transpose(host_force, (1, 2, 3, 0))
    print(host_force.shape)
    print(f_1.shape)
    force = wp.from_numpy(host_force, dtype=precision_policy.store_precision.wp_dtype)

    # ----------define exact solution-----------
    exact_u = lambda x, y: cos(x)
    exact_v = lambda x, y: cos(y)

    for i in range(timesteps):
        wp.synchronize() #probablynot needed according to docs (automatic synchronization between kernel launches)
        wp.launch(
            collide,
            inputs=[f_1, f_2, force, displacement, omega, theta],
            dim=f_1.shape[1:],
        )
        wp.launch(stream, inputs=[f_2, f_1, nodes_x, nodes_x], dim=f_1.shape[1:])
        wp.synchronize()
        if i%5 == 0: post_process(displacement, L, T, kappa)
