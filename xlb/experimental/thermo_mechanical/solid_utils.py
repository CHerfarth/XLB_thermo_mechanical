import warp as wp
from xlb.precision_policy import PrecisionPolicy
from typing import Any
import sympy
import numpy as np
from xlb.utils import save_fields_vtk, save_image

np.seterr(all="ignore")


# Mapping:
#    i  j   |   f_q
#    0  0   |   0
#    0  1   |   1
#   0  -1   |   2
#    1  0   |   3
#   -1  1   |   4
#    1  -1  |   5
#   -1  0   |   6
#    1  1   |   7
#   -1  -1  |   8

# Mapping for moments:
#    i  j   |   m_q
#    1  0   |   0
#    0  1   |   1
#    1  1   |   2
#    s      |   3
#    d      |   4
#    1  2   |   5
#    2  1   |   6
#    2  2   |   7
#    0  0   |   8 (irrelevant)


solid_vec = wp.vec(
    9, dtype=PrecisionPolicy.FP32FP32.compute_precision.wp_dtype
)  # this is the default precision policy; it can be changed by calling set_precision_policy()


def set_precision_policy(precision_policy):
    global solid_vec
    solid_vec = wp.vec(9, dtype=precision_policy.FP32FP32.compute_precision.wp_dtype)


@wp.func
def read_local_population(f: wp.array4d(dtype=Any), x: wp.int32, y: wp.int32):
    f_local = solid_vec()
    for i in range(9):
        f_local[i] = f[i, x, y, 0]
    return f_local


@wp.func
def write_population_to_global(f: wp.array4d(dtype=Any), f_local: solid_vec, x: wp.int32, y: wp.int32):
    for i in range(9):
        f[i, x, y, 0] = f_local[i]


@wp.func
def write_vec_to_global(array: wp.array4d(dtype=Any), array_local: Any, x: wp.int32, y: wp.int32, dim: wp.int32):
    for i in range(dim):
        array[i, x, y, 0] = array_local[i]


@wp.func
def calc_moments(f: solid_vec):
    m = solid_vec()
    # Todo: find better way to do this!
    m[0] = f[3] - f[6] + f[7] - f[4] - f[8] + f[5]
    m[1] = f[1] - f[2] + f[7] + f[4] - f[8] - f[5]
    m[2] = f[7] - f[4] + f[8] - f[5]
    m[3] = f[3] + f[1] + f[6] + f[2] + 2.0 * f[7] + 2.0 * f[4] + 2.0 * f[8] + 2.0 * f[5]
    m[4] = f[3] - f[1] + f[6] - f[2]
    m[5] = f[7] - f[4] - f[8] + f[5]
    m[6] = f[7] + f[4] - f[8] - f[5]
    m[7] = f[7] + f[4] + f[8] + f[5]
    m[8] = 0.0
    return m


@wp.kernel
def copy_populations(origin: wp.array4d(dtype=Any), dest: wp.array4d(dtype=Any), dim: Any):
    i, j, k = wp.tid()
    for l in range(dim):
        dest[l, i, j, 0] = origin[l, i, j, 0]


@wp.func
def calc_populations(m: solid_vec):
    f = solid_vec()
    # Todo: find better way to do this!
    f[3] = 2.0 * m[0] + m[3] + m[4] - 2.0 * m[5] - 2.0 * m[7]
    f[1] = 2.0 * m[1] + m[3] - m[4] - 2.0 * m[6] - 2.0 * m[7]
    f[6] = -2.0 * m[0] + m[3] + m[4] + 2.0 * m[5] - 2.0 * m[7]
    f[2] = -2.0 * m[1] + m[3] - m[4] + 2.0 * m[6] - 2.0 * m[7]
    f[7] = m[2] + m[5] + m[6] + m[7]
    f[4] = -m[2] - m[5] + m[6] + m[7]
    f[8] = m[2] - m[5] - m[6] + m[7]
    f[5] = -m[2] + m[5] - m[6] + m[7]
    f[0] = 0.0
    for i in range(9):
        f[i] = f[i] / 4.0
    return f


@wp.func
def calc_equilibrium(m: solid_vec, theta: Any):
    m_eq = solid_vec()
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


def get_force_load(manufactured_displacement, x, y, mu, K):
    man_u = manufactured_displacement[0]
    man_v = manufactured_displacement[1]
    b_x = -mu * (sympy.diff(man_u, x, x) + sympy.diff(man_u, y, y)) - K * sympy.diff(sympy.diff(man_u, x) + sympy.diff(man_v, y), x)
    b_y = -mu * (sympy.diff(man_v, x, x) + sympy.diff(man_v, y, y)) - K * sympy.diff(sympy.diff(man_u, x) + sympy.diff(man_v, y), y)
    return (np.vectorize(sympy.lambdify([x, y], b_x, "numpy")), np.vectorize(sympy.lambdify([x, y], b_y, "numpy")))


def get_function_on_grid(f, x, y, dx, grid):
    f = np.vectorize(sympy.lambdify([x, y], f, "numpy"))
    f_scaled = lambda x_node, y_node: f((x_node + 0.5) * dx, (y_node + 0.5) * dx)
    f_on_grid = np.fromfunction(f_scaled, shape=grid.shape)
    return f_on_grid


def get_error_norms(current_macroscopics, expected_macroscopics, dx, timestep=0):
    error_matrix = np.subtract(current_macroscopics[0:4, :, :, 0], expected_macroscopics[0:4, :, :])
    """print("----------CURRENT-----------")
    print(current_macroscopics)
    print("____________EXPECTED______________")
    print(expected_macroscopics)
    print("__________ERROR____________")
    print(error_matrix)"""
    # step 1: handle displacement
    l2_disp = np.sqrt(np.nansum(np.linalg.norm(error_matrix[0:2, :, :], axis=0) ** 2)) * dx
    linf_disp = np.nanmax(np.nan_to_num(np.max(np.abs(error_matrix[0:2, :, :]), axis=0)))
    # step 2: handle stress
    l2_stress = np.sqrt(np.nansum(np.linalg.norm(error_matrix[2:5, :, :], axis=0) ** 2)) * dx
    linf_stress = np.nanmax(np.max(np.abs(error_matrix[2:5, :, :]), axis=0))
    # step 3: output error image
    # error_inf = np.nanmax(np.abs(error_matrix[2:4, :, :]), axis=0)
    # fields = {"error_inf": error_inf}
    # save_fields_vtk(fields, timestep=timestep, prefix="error")
    return l2_disp, linf_disp, l2_stress, linf_stress


def get_expected_stress(manufactured_displacement, x, y, lamb, mu):
    man_u = manufactured_displacement[0]
    man_v = manufactured_displacement[1]
    e_xx = sympy.diff(man_u, x)
    e_yy = sympy.diff(man_v, y)
    e_xy = 0.5 * (sympy.diff(man_u, y) + sympy.diff(man_v, x))
    s_xx = lamb * (e_xx + e_yy) + 2 * mu * e_xx
    s_yy = lamb * (e_xx + e_yy) + 2 * mu * e_yy
    s_xy = 2 * mu * e_xy
    return s_xx, s_yy, s_xy


def restrict_solution_to_domain(array, potential, dx):  # ToDo: make more efficient (fancy numpy funcs)
    if potential == None:
        return array
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            if potential(i * dx + 0.5 * dx, j * dx + 0.5 * dx) > 0:
                array[:, i, j] = np.nan
    return array


def output_image(macroscopics, timestep, name, potential=None, dx=None):
    dis_x = macroscopics[0, :, :, 0]
    dis_y = macroscopics[1, :, :, 0]
    s_xx = macroscopics[2, :, :, 0]
    s_yy = macroscopics[3, :, :, 0]
    s_xy = macroscopics[4, :, :, 0]
    # output as vtk files
    dis_mag = np.sqrt(np.square(dis_x) + np.square(dis_y))
    fields = {"dis_x": dis_x, "dis_y": dis_y, "dis_mag": dis_mag, "s_xx": s_xx, "s_yy": s_yy, "s_xy": s_xy}
    save_fields_vtk(fields, timestep=timestep, prefix=name)
    save_image(dis_mag, timestep)


def process_error(macroscopics, expected_macroscopics, timestep, dx, norms_over_time):
    # calculate error to expected solution
    l2_disp, linf_disp, l2_stress, linf_stress = get_error_norms(macroscopics, expected_macroscopics, dx, timestep)
    norms_over_time.append((timestep, l2_disp, linf_disp, l2_stress, linf_stress))
    return l2_disp, linf_disp, l2_stress, linf_stress
