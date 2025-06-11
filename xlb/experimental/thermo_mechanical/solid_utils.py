import warp as wp
from xlb.precision_policy import PrecisionPolicy
from typing import Any
import sympy
import numpy as np
from xlb.utils import save_fields_vtk, save_image
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import matplotlib.pyplot as plt
import statistics


def get_force_load(manufactured_displacement, x, y):
    params = SimulationParams()
    mu = params.mu_unscaled
    K = params.K_unscaled
    man_u = manufactured_displacement[0]
    man_v = manufactured_displacement[1]
    b_x = -mu * (sympy.diff(man_u, x, x) + sympy.diff(man_u, y, y)) - K * sympy.diff(
        sympy.diff(man_u, x) + sympy.diff(man_v, y), x
    )
    b_y = -mu * (sympy.diff(man_v, x, x) + sympy.diff(man_v, y, y)) - K * sympy.diff(
        sympy.diff(man_u, x) + sympy.diff(man_v, y), y
    )
    return (
        np.vectorize(sympy.lambdify([x, y], b_x, "numpy")),
        np.vectorize(sympy.lambdify([x, y], b_y, "numpy")),
    )


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


def rmsd(array):
    vector = array.flatten
    n = vector.size()
    return math.sqrt((1 / n) * np.linalg.norm(vector))


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


def restrict_solution_to_domain(
    array, potential, dx
):  # ToDo: make more efficient (fancy numpy funcs)
    if potential == None:
        return array
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            if potential(i * dx + 0.5 * dx, j * dx + 0.5 * dx) > 0:
                array[:, i, j] = np.nan
    return array


def output_image(macroscopics, timestep, name, potential=None):
    dis_x = macroscopics[0, :, :, 0]
    dis_y = macroscopics[1, :, :, 0]
    s_xx = macroscopics[2, :, :, 0]
    s_yy = macroscopics[3, :, :, 0]
    s_xy = macroscopics[4, :, :, 0]
    # output as vtk files
    dis_mag = np.sqrt(np.square(dis_x) + np.square(dis_y))
    fields = {
        "dis_x": dis_x,
        "dis_y": dis_y,
        "dis_mag": dis_mag,
        "s_xx": s_xx,
        "s_yy": s_yy,
        "s_xy": s_xy,
    }
    save_fields_vtk(fields, timestep=timestep, prefix=name)
    save_image(dis_mag, timestep)


def process_error(macroscopics, expected_macroscopics, timestep, dx, norms_over_time):
    # calculate error to expected solution
    l2_disp, linf_disp, l2_stress, linf_stress = get_error_norms(
        macroscopics, expected_macroscopics, dx, timestep
    )
    norms_over_time.append((timestep, l2_disp, linf_disp, l2_stress, linf_stress))
    return l2_disp, linf_disp, l2_stress, linf_stress


def get_initial_guess_from_white_noise(shape, precision_policy, dx, mean=0, seed=31):
    kernel_provider = KernelProvider()
    convert_moments_to_populations = kernel_provider.convert_moments_to_populations

    params = SimulationParams()
    theta = params.theta
    dx = params.dx
    mu = params.mu_unscaled
    lamb = params.lamb_unscaled
    L = params.L
    T = params.T
    kappa = params.kappa

    rng = np.random.default_rng(seed)

    # create white noise array on host
    host = rng.normal(loc=mean, scale=1.0, size=shape)
    host[2:, :, :, :] = np.zeros_like(host[2:, :, :, :])
    #calculate infinitesimal strain tensor
    u_x = host[0,:,:,0]
    u_y = host[0,:,:,0]
    e_xx = np.gradient(u_x, dx, axis=0)
    e_yy = np.gradient(u_y, dx, axis=1)
    dux_dy = np.gradient(u_x, dx, axis=1)
    duy_dx = np.gradient(u_y, dx, axis=0)
    e_xy = 0.5*(dux_dy + duy_dx)

    s_xx = lamb*(e_xx + e_yy) + 2*mu*e_xx
    s_yy = lamb*(e_xx + e_yy) + 2*mu*e_yy
    s_xy = 2*mu*e_xy

    s_s = s_xx + s_yy
    s_d = s_xx - s_yy
    
    s_s = s_s #* T / (L*kappa)
    s_d = s_d #* T /(L*kappa)
    s_xy = s_xy# * T / (L*kappa)
    #set all other moments consistent to first order moments
    mu = params.mu
    lamb = params.lamb
    K = params.K

    tau_11 = mu/theta
    tau_s = 2*K/(1+theta)
    tau_d = 2*mu/(1-theta)
    tau_f = 0.5


    host[2, :, :, 0] = -(1 + 1/(2*tau_11))*s_xy
    host[3,:,:,0] = -(1+1/tau_s)*s_s
    host[4,:,:,0] = -(1+1/tau_d)*s_d
    host[5,:,:,0] = theta*u_x
    host[6,:,:,0] = theta*u_y
    host[7,:,:,0] = ((-theta*(1+2*tau_f))/((1+theta)*(tau_s-tau_f)))*s_s

    # manually set to expected mean
    for l in range(2):
        host[l, :, :, 0] = host[l, :, :, 0] - np.full(
            shape=host[l, :, :, 0].shape, fill_value=(np.sum(host[l, :, :, 0]/(host.shape[1]*host.shape[2])) - mean)
        )
    
    print(np.sum(host[0,:,:,0]))
    print(np.sum(host[1,:,:,0]))

    # load onto device
    device = wp.from_numpy(host, dtype=precision_policy.store_precision.wp_dtype)

    # convert to populations
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


def plot_3d_wireframe(data, name="wireframe", timestep=0, stride=5, zlim=(-1, 1)):
    """Create 3D wireframe plot"""
    output_file = name + "_" + str(timestep) + ".png"
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Create coordinate arrays
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create wireframe plot
    ax.plot_wireframe(X, Y, data, rstride=stride, cstride=stride, alpha=0.7)
    ax.set_zlim(zlim)

    ax.set_xlabel("X Index")
    ax.set_ylabel("Y Index")
    ax.set_zlabel("Value")
    ax.set_title("3D Wireframe Plot")

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_3d_surface(data, name="surface", timestep=0, colormap="viridis", zlim=None):
    """Create 3D surface plot"""
    output_file = name + "_" + str(timestep) + ".png"
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Create coordinate arrays
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create surface plot
    if zlim != None:
        surf = ax.plot_surface(
            X,
            Y,
            data,
            cmap=colormap,
            vmin=zlim[0],
            vmax=zlim[1],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )
        ax.set_zlim(zlim)
    else:
        surf = ax.plot_surface(X, Y, data, cmap=colormap, alpha=0.9, linewidth=0, antialiased=True)

    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=20)

    ax.set_xlabel("X Index")
    ax.set_ylabel("Y Index")
    ax.set_zlabel("Value")
    ax.set_title("3D Surface Plot")

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_x_slice(
    array1,
    dx1,
    array2=None,
    dx2=None,
    zlim=None,
    name="slice",
    timestep=0,
    y_index=None,
    xlabel="x",
    ylabel="val",
    title="Slice along x-Axis",
    label1="Array 1",
    label2="Array 2",
):
    """
    Plots a slice of a 2D array along the x-axis at a given y_index.
    If y_index is None, uses the middle row.
    """
    output_file = name + "_" + str(timestep) + ".png"
    if y_index is None:
        y_index_1 = array1.shape[0] // 2  # Middle row
        if dx2 != None:
            y_index_2 = array2.shape[0] // 2  # Middle row

    plt.figure()
    x1 = np.arange(array1.shape[1]) * dx1 + 0.5 * dx1
    plt.plot(x1, array1[y_index_1, :], "-d", label=label1)

    if dx2 != None:
        x2 = np.arange(array2.shape[1]) * dx2 + 0.5 * dx2
        plt.plot(x2, array2[y_index_2, :], "-s", label=label2)

    if dx2 == dx1:
        plt.plot(x1, (array1 + array2)[y_index_1, :], "-o", label="combined+")
        # plt.plot(x1, (array1 - array2)[y_index_1, :], '-o', label='combined-')

    if zlim != None:
        plt.ylim(zlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (y={y_index})")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def rate_of_convergence(data, column_header, min=None, max=None):
    if min is not None and max is not None:
        filtered = [row for row in data[column_header] if row > min and row < max]
    elif min is not None:
        filtered = [row for row in data[column_header] if row > min]
    elif max is not None:
        filtered = [row for row in data[column_header] if row < max]
    else:
        filtered = [row for row in data[column_header]]

    rates = list()
    for i in range(len(filtered) - 1):
        rates.append(filtered[i + 1] / filtered[i])
    return statistics.median(rates)
