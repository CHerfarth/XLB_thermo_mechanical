import numpy as np
import warp as wp
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.experimental.thermo_mechanical.solid_utils as utils
import math
from typing import Any


class Level:
    def __init__(self, nodes_x, nodes_y, dx, dt, force_load, gamma, compute_backend, velocity_set, precision_policy):
        self.precision_policy = precision_policy
        self.grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.velocity_set = velocity_set
        c = self.velocity_set.c_float
        # params needed to set up simulation params
        self.gamma = gamma
        self.dx = dx
        self.dt = dt
        self.startup()
        # setup grids
        self.f_1 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_2 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_3 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_4 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.macroscopics = self.grid.create_field(cardinality=5, dtype=precision_policy.store_precision)  # 5 macroscopic variables
        self.residual = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.defect_correction = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        # setup stepper
        self.stepper = SolidsStepper(self.grid, force_load)

        @wp.kernel
        def relaxation(f_after_stream: Any, f_previous: Any, defect_correction: Any, f_destination: Any, gamma: wp.float32):
            i, j, k = wp.tid()
            for l in range(velocity_set.q):
                f_destination[l, i, j, 0] = (
                    gamma * (f_after_stream[l, i, j, 0] + 0.*defect_correction[l, i, j, 0]) + (1.0 - gamma) * f_previous[l, i, j, 0]
                )

        self.relax = relaxation

        @wp.kernel
        def set_from_macroscopics(macroscopics: Any, f: Any, force: Any, theta: Any, K: Any, mu: Any, L: Any, T: Any):
            i, j, k = wp.tid()
            u_x = macroscopics[0, i, j, 0]
            u_y = macroscopics[1, i, j, 0]
            s_xx = macroscopics[2, i, j, 0] * T / L
            s_yy = macroscopics[3, i, j, 0] * T / L
            s_xy = macroscopics[4, i, j, 0] * T / L
            # printf("I: %d, J: %d, ux: %f, uy: %f\n", i, j, u_x, u_y)
            s_s = s_xx + s_yy
            s_d = s_xx - s_yy
            # set populations
            for l in range(9):
                x_dir = c[0, l]
                y_dir = c[1, l]
                if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 1.0) < 1e-3:  # case V1
                    f[l, i, j, 0] = (1.0 - theta) * 0.5 * (x_dir * (u_x - 0.5 * force[0, i, j, 0]) + y_dir * (u_y - 0.5 * force[1, i, j, 0]))
                    #f[l, i, j, 0] += -(1.0 - theta + 4.0 * K) * s_s / (8.0 * K)
                    #f[l, i, j, 0] += -x_dir * y_dir * (1.0 - theta - 4.0 * mu) * s_d / (8.0 * mu)
                if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 2.0) < 1e-3:  # case V2
                    f[l, i, j, 0] = 0.25 * theta * (x_dir * (u_x) + y_dir * (u_y))
                    #f[l, i, j, 0] += theta * s_s / (8.0 * K)  # careful! changed sign compared to paper
                    #f[l, i, j, 0] += -x_dir * y_dir * (theta + 2.0 * mu) * s_xy / (8.0 * mu)

        self.set_from_macroscopics = set_from_macroscopics

    def startup(self):
        solid_simulation = SimulationParams()
        E = solid_simulation.E_unscaled
        nu = solid_simulation.nu_unscaled
        kappa = solid_simulation.kappa
        theta = solid_simulation.theta
        solid_simulation.set_parameters(E=E, nu=nu, dx=self.dx, dt=self.dt, L=self.dx, T=self.dt, kappa=kappa, theta=theta)

    def init_from_macroscopics(self, macroscopics, destination):
        self.startup()
        solid_simulation = SimulationParams()
        theta = solid_simulation.theta
        K = solid_simulation.K
        mu = solid_simulation.mu
        L = solid_simulation.L
        T = solid_simulation.T
        wp.launch(self.set_from_macroscopics, inputs=[macroscopics, destination, self.stepper.force, theta, K, mu, L, T], dim=self.f_1.shape[1:])

    def set_defect_correction(self):
        self.stepper(self.f_4, self.f_3)  # perform one step of operator on restricted finer grid approximation
        self.f_4, self.f_3 = self.f_3, self.f_4
        wp.launch(utils.add_populations, inputs=[self.f_4, self.residual, self.defect_correction, 9], dim=self.f_4.shape[1:])
    
    def set_to_zero(self):
        self.f_1 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        self.f_2 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        self.f_3 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        self.f_4 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        self.macroscopics = self.grid.create_field(cardinality=5, dtype=self.precision_policy.store_precision)  # 5 macroscopic variables
        self.residual = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        self.defect_correction = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)


    def get_error_approx(self):
        wp.launch(utils.subtract_populations, inputs=[self.f_1, self.f_4, self.f_3, 9], dim=self.f_1.shape[1:])
        return self.f_3

    def perform_smoothing(self, get_residual=False):
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_2)
        #self.f_1, self.f_2 = self.f_2, self.f_1
        # wp.launch(utils.add_populations, inputs=[self.f_2, self.defect_correction, self.f_2, 9], dim=self.f_1.shape[1:])
        wp.launch(self.relax, inputs=[self.f_2, self.f_3, self.defect_correction, self.f_1, self.gamma], dim=self.f_1.shape[1:])

        if get_residual:
            wp.launch(utils.get_residual, inputs=[self.f_3, self.f_1, self.residual, self.velocity_set.q], dim=self.f_1.shape[1:])

    def get_macroscopics(self):
        self.startup()
        return self.stepper.get_macroscopics(self.f_1)
    
    def get_macroscopics_device(self):
        self.startup()
        return self.stepper.get_macroscopics_device(self.f_1)

    def info(self):
        """
        Print detailed information about this Level.
        """
        print("\n===== Level Information =====")
        print(f"Grid Size: {self.nodes_x} x {self.nodes_y} nodes")
        print(f"Physical Parameters:")
        print(f"  - dx: {self.dx}")
        print(f"  - dt: {self.dt}")
        print(f"  - gamma (relaxation parameter): {self.gamma}")

        # Get velocity set information
        print(f"Velocity Set: {self.velocity_set.__class__.__name__}")
        print(f"  - q (number of velocities): {self.velocity_set.q}")

        # Field information
        print("\nFields:")
        print(f"  - f_1 shape: {self.f_1.shape}")
        print(f"  - f_2 shape: {self.f_2.shape}")
        print(f"  - f_3 shape: {self.f_3.shape}")
        print(f"  - f_4 shape: {self.f_4.shape}")
        print(f"  - residual shape: {self.residual.shape}")
        print(f"  - defect_correction shape: {self.defect_correction.shape}")

        # Stepper information
        print("\nStepper Information:")
        print(f"  - Type: {self.stepper.__class__.__name__}")

        # Try to get parameters from the SimulationParams
        try:
            solid_simulation = SimulationParams()
            E = solid_simulation.E_unscaled
            nu = solid_simulation.nu_unscaled
            kappa = solid_simulation.kappa
            theta = solid_simulation.theta
            print("\nSimulation Parameters:")
            print(f"  - E (Young's modulus): {E}")
            print(f"  - nu (Poisson ratio): {nu}")
            print(f"  - kappa: {kappa}")
            print(f"  - theta: {theta}")
        except Exception as e:
            print(f"\nCould not retrieve simulation parameters: {e}")

        print("============================\n")


@wp.kernel
def interpolate(coarse: Any, fine: Any, nodes_x_coarse: wp.int32, nodes_y_coarse: wp.int32, dim: Any):
    i, j, k = wp.tid()
    i_fine = 2 * i
    j_fine = 2 * j

    nodes_x_fine = nodes_x_coarse * 2
    nodes_y_fine = nodes_y_coarse * 2

    for l in range(dim):
        fine[l, i_fine, j_fine, 0] = coarse[l, i, j, 0]
        fine[l, wp.mod(i_fine + 1, nodes_x_fine), j_fine, 0] = 0.5 * (coarse[l, i, j, 0] + coarse[l, wp.mod(i + 1, nodes_x_coarse), j, 0])
        fine[l, i_fine, wp.mod(j_fine + 1, nodes_y_fine), 0] = 0.5 * (coarse[l, i, j, 0] + coarse[l, i, wp.mod(j + 1, nodes_y_coarse), 0])
        fine[l, wp.mod(i_fine + 1, nodes_x_fine), wp.mod(j_fine + 1, nodes_y_fine), 0] = 0.25 * (
            coarse[l, i, j, 0]
            + coarse[l, wp.mod(i + 1, nodes_x_coarse), j, 0]
            + coarse[l, i, wp.mod(j + 1, nodes_y_coarse), 0]
            + coarse[l, wp.mod(i + 1, nodes_x_coarse), wp.mod(j + 1, nodes_y_coarse), 0]
        )


@wp.kernel
def restrict(coarse: Any, fine: Any, dim: Any):
    i, j, k = wp.tid()
    for l in range(dim):
        coarse[l, i, j, 0] = fine[l, 2 * i, 2 * j, 0]


class MultigridSolver:
    """
    A class implementing a multigrid iterative solver for elliptic PDEs.
    """

    def __init__(self, nodes_x, nodes_y, length_x, length_y, dt, E, nu, force_load, gamma, timesteps, max_levels=None):
        compute_backend = ComputeBackend.WARP
        precision_policy = PrecisionPolicy.FP32FP32
        velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)
        xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

        solid_simulation = SimulationParams()
        solid_simulation.set_parameters(
            E=E, nu=nu, dx=1, dt=1, L=1, T=1, kappa=1, theta=1.0 / 3.0
        )  # just placeholder, so E and nu get past to all levels
        self.timesteps = timesteps

        # TODO: boundary conditions

        # Determine maximum possible levels
        self.max_possible_levels = min(int(np.log2(nodes_x - 1)), int(np.log2(nodes_y - 1))) + 1

        if max_levels is None:
            self.max_levels = self.max_possible_levels
        else:
            self.max_levels = min(max_levels, self.max_possible_levels)

        # setup levels
        self.levels = list()
        self.iterations_on_level = list()
        for i in range(self.max_levels):
            nx_level = int((nodes_x) / (2**i))  # IMPORTANT: only works with nodes as power of two at the moment
            ny_level = int((nodes_y) / (2**i))
            dx = length_x / float(nx_level)
            dy = length_y / float(ny_level)
            dt_level = dt * (2**i) #!!!
            print("Level: {}, dx: {}, dt: {}".format(i, dx, dt_level))
            assert math.isclose(dx, dy)
            level = Level(
                nodes_x=nx_level,
                nodes_y=ny_level,
                dx=dx,
                dt=dt_level,
                force_load=force_load,
                gamma=gamma,
                compute_backend=compute_backend,
                velocity_set=velocity_set,
                precision_policy=precision_policy,
            )
            self.levels.append(level)
            self.iterations_on_level.append(2**(self.max_levels - i + 1))
        self.current_level_num = self.max_levels - 1
        self.current_iterations = 0

    def get_macroscopics_on_finest_grid(self):
        level_num = self.current_level_num
        while level_num > 0:
            coarse_level = self.levels[level_num]
            fine_level = self.levels[level_num - 1]
            coarse_macroscopics = coarse_level.get_macroscopics_device()
            # Interpolate the macroscopics from the coarse level to the fine level
            wp.launch(
                interpolate,
                inputs=[coarse_macroscopics, fine_level.macroscopics, coarse_level.nodes_x, coarse_level.nodes_y, 5],
                dim=coarse_level.macroscopics.shape[1:],
            )
            fine_level.init_from_macroscopics(fine_level.macroscopics, fine_level.f_1)
            level_num -= 1
        # Now we are on the finest level
        macroscopics = self.levels[0].stepper.get_macroscopics(self.levels[0].f_1)
        return macroscopics

    def work(self, timestep, return_macroscopics=False):
        if timestep != 0 and (self.current_iterations - self.iterations_on_level[self.current_level_num]) == 0 and self.current_level_num != 0:  # transfer to lower grid
            previous_level = self.levels[self.current_level_num]
            self.current_level_num += -1
            current_level = self.levels[self.current_level_num]
            previous_macroscopics = previous_level.get_macroscopics_device()
            # Interpolate the macroscopics from the previous level to the current level
            wp.launch(
                interpolate,
                inputs=[previous_macroscopics, current_level.macroscopics, previous_level.nodes_x, previous_level.nodes_y, 5],
                dim=previous_level.macroscopics.shape[1:],
            )
            current_level.init_from_macroscopics(current_level.macroscopics, current_level.f_1)
            self.current_iterations = 0
            print("Timestep: {}, Changing to Level: {}".format(timestep, self.current_level_num))
        level = self.levels[self.current_level_num]
        level.perform_smoothing()
        self.current_iterations += 1
        if return_macroscopics:
            return self.get_macroscopics_on_finest_grid()