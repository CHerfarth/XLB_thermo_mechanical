import xlb
import sys
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
import xlb.experimental
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.velocity_set
import warp as wp
import numpy as np
from typing import Any
import sympy
import csv
import math
import xlb.experimental.thermo_mechanical.solid_utils as utils
import xlb.experimental.thermo_mechanical.solid_bounceback as bc
from xlb.utils import save_fields_vtk, save_image
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider


def write_results(norms_over_time, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "L2", "Linf"])
        writer.writerows(norms_over_time)


if __name__ == "__main__":
    wp.config.mode = "debug"
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, compute_backend=compute_backend
    )

    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    # initialize grid
    nodes_x = 128
    nodes_y = 128
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)

    # get discretization
    length_x = 1
    length_y = 1
    dx = length_x / float(nodes_x)
    dy = length_y / float(nodes_y)
    assert math.isclose(dx, dy)
    timesteps = 50
    dt = dx * dx

    # params
    E = 0.5
    nu = 0.5

    solid_simulation = SimulationParams()
    solid_simulation.set_all_parameters(
        E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1, theta=1.0 / 3.0
    )
    print("E: {}, nu: {}".format(solid_simulation.E, solid_simulation.nu))

    # get force load
    x, y = sympy.symbols("x y")
    manufactured_u = 0 * x  # sympy.cos(2 * sympy.pi * x)*sympy.sin(6*sympy.pi*y)
    manufactured_v = 0 * y  # sympy.cos(2 * sympy.pi * y)*sympy.sin(8*sympy.pi*x)
    expected_displacement = np.array([
        utils.get_function_on_grid(manufactured_u, x, y, dx, grid),
        utils.get_function_on_grid(manufactured_v, x, y, dx, grid),
    ])
    force_load = utils.get_force_load((manufactured_u, manufactured_v), x, y)

    # get expected stress
    s_xx, s_yy, s_xy = utils.get_expected_stress((manufactured_u, manufactured_v), x, y)
    expected_stress = np.array([
        utils.get_function_on_grid(s_xx, x, y, dx, grid),
        utils.get_function_on_grid(s_yy, x, y, dx, grid),
        utils.get_function_on_grid(s_xy, x, y, dx, grid),
    ])

    # set boundary potential
    potential_sympy = (0.5 - x) ** 2 + (0.5 - y) ** 2 - 0.2
    potential = sympy.lambdify([x, y], potential_sympy)
    indicator = lambda x, y: -1
    boundary_array, boundary_values = bc.init_bc_from_lambda(
        potential_sympy, grid, dx, velocity_set, (manufactured_u, manufactured_v), indicator, x, y
    )

    """#--------------------do with dirichlet bc----------------------------------
    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values)

    # startup grids
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_3 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)

    norms_over_time = list()  # to track error over time
    for i in range(timesteps):
        stepper(f_1, f_2)
        f_1, f_2 = f_2, f_1
        macroscopics = stepper.get_macroscopics_host(f_1)
        l2_disp, l2_inf, l2_stress, linf_stress = utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
        if i % 1 == 0:
            macroscopics = stepper.get_macroscopics_host(f_1)
            utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
            utils.output_image(macroscopics, i, "dirichlet", potential)"""

    # --------------------do with standard----------------------------------
    kernel_provider = KernelProvider()
    copy_populations = kernel_provider.copy_populations
    multiply_populations = kernel_provider.multiply_populations
    add_populations = kernel_provider.add_populations
    subtract_populations = kernel_provider.subtract_populations
    l2_norm_squared = kernel_provider.l2_norm
    relaxation_no_defect = kernel_provider.relaxation_no_defect

    potential, boundary_array, boundary_values = None, None, None
    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(
        grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values
    )

    # startup grids
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    residual = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    # set initial guess from white noise
    f_1 = utils.get_initial_guess_from_white_noise(f_2.shape, precision_policy, dx, mean=0, seed=31)

    norms_over_time = list()  # to track error over time
    gamma = 0.8
    for i in range(timesteps):
        if i % 1 == 0:
            macroscopics = stepper.get_macroscopics_host(f_1)
            utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
            utils.output_image(macroscopics, i, "standard", potential)
        # wp.launch(copy_populations, inputs=[f_1, residual, 9], dim=f_1.shape[1:])
        stepper(f_1, f_2)
        f_1, f_2 = f_2, f_1
        # wp.launch(relaxation_no_defect, inputs=[f_2, residual, f_1, gamma, 9], dim=f_2.shape[1:])

    # --------------------do with relaxed----------------------------------
    kernel_provider = KernelProvider()
    copy_populations = kernel_provider.copy_populations
    multiply_populations = kernel_provider.multiply_populations
    add_populations = kernel_provider.add_populations
    subtract_populations = kernel_provider.subtract_populations
    l2_norm_squared = kernel_provider.l2_norm
    relaxation_no_defect = kernel_provider.relaxation_no_defect

    potential, boundary_array, boundary_values = None, None, None
    # adjust expected solution
    expected_macroscopics = np.concatenate((expected_displacement, expected_stress), axis=0)
    expected_macroscopics = utils.restrict_solution_to_domain(expected_macroscopics, potential, dx)

    # initialize stepper
    stepper = SolidsStepper(
        grid, force_load, boundary_conditions=boundary_array, boundary_values=boundary_values
    )

    # startup grids
    f_2 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    residual = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    # set initial guess from white noise
    f_1 = utils.get_initial_guess_from_white_noise(f_2.shape, precision_policy, dx, mean=0, seed=31)

    norms_over_time = list()  # to track error over time
    gamma = 0.8
    for i in range(timesteps):
        if i % 1 == 0:
            macroscopics = stepper.get_macroscopics_host(f_1)
            utils.process_error(macroscopics, expected_macroscopics, i, dx, norms_over_time)
            utils.output_image(macroscopics, i, "relaxed", potential)
        wp.launch(copy_populations, inputs=[f_1, residual, 9], dim=f_1.shape[1:])
        stepper(f_1, f_2)
        wp.launch(relaxation_no_defect, inputs=[f_2, residual, f_1, gamma, 9], dim=f_2.shape[1:])
