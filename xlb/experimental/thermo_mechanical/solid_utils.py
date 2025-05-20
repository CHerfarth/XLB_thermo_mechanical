import warp as wp
from xlb.precision_policy import PrecisionPolicy
from typing import Any
import sympy
import numpy as np
from xlb.utils import save_fields_vtk, save_image
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider


def get_force_load(manufactured_displacement, x, y):
    params = SimulationParams()
    mu = params.mu_unscaled
    K = params.K_unscaled
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


def get_expected_stress(manufactured_displacement, x, y):
    params = SimulationParams()
    lamb = params.lamb_unscaled  # unscaled, because we want the expected stress in normal unit
    mu = params.mu_unscaled
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


def get_initial_guess_from_white_noise(shape, precision_policy, dx, mean=0, seed=31):
    kernel_provider = KernelProvider()
    convert_moments_to_populations = kernel_provider.convert_moments_to_populations

    rng = np.random.default_rng(seed)

    #create white noise array on host 
    host = rng.normal(loc=mean, scale=1.0, size=shape)

    host[2:,:,:,:] = np.zeros_like(host[2:,:,:,:])
    #manually set to expected mean
    for l in range(2):
        host[l,:,:,0] = host[l,:,:,0] - np.full(shape=host[l,:,:,0].shape, fill_value=(np.sum(host[l,:,:,0])*dx*dx - mean))

    #load onto device
    device = wp.from_numpy(host, dtype=precision_policy.store_precision.wp_dtype)

    #convert to populations
    wp.launch(convert_moments_to_populations, inputs=[device, device], dim=device.shape[1:])

    return device


def last_n_avg(data, n):
    length = len(data)
    weight = 1 / min(n, length)
    val = 0.0
    for i in range(min(n, length)):
        val += data[length - 1 - i]
    val = val * weight
    return val

