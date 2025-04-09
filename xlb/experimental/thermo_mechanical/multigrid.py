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
                    gamma * (f_after_stream[l, i, j, 0] - defect_correction[l, i, j, 0]) + (1.0 - gamma) * (f_previous[l, i, j, 0])
                )

        self.relax = relaxation

        @wp.kernel
        def set_from_macroscopics(macroscopics: Any, f_fine: Any, theta: Any, K: Any, mu: Any, L: Any, T: Any):
            i, j, k = wp.tid()
            u_x = macroscopics[0,i,j,0] 
            u_y = macroscopics[1,i,j,0] 
            s_xx = macroscopics[2,i,j,0] * T / L
            s_yy = macroscopics[3,i,j,0] * T / L
            s_xy = macroscopics[4,i,j,0] * T / L
            #printf("I: %d, J: %d, ux: %f, uy: %f\n", i, j, u_x, u_y)
            s_s = s_xx + s_yy   
            s_d = s_xx - s_yy
            #set populations
            for l in range(9):
                x_dir = c[0, l]
                y_dir = c[1, l]
                #printf("l: %d, x_dir: %f, y_dir: %f\n", l, x_dir, y_dir)
                if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 1.0) < 1e-3: #case V1
                    f_fine[l, i, j, 0] = (1.-theta)*0.5*(x_dir*u_x + y_dir*u_y)
                    #f_fine[l, i, j, 0] += -(1.-theta+4.*K)*s_s/(8.*K)
                    #f_fine[l, i, j, 0] += -x_dir*y_dir*(1.-theta-4.*mu)*s_d/(8.*mu)
                if wp.abs(wp.abs(x_dir) + wp.abs(y_dir) - 2.0) < 1e-3: #case V2
                    f_fine[l, i, j, 0] = 0.25*theta*(x_dir*u_x + y_dir*u_y)
                    #f_fine[l, i, j, 0] += -theta*s_s/(8.*K)
                    #f_fine[l, i, j, 0] += -x_dir*y_dir*(theta+2.*mu)*s_xy/(8.*mu)

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
        wp.launch(self.set_from_macroscopics, inputs=[macroscopics, destination, theta, K, mu, L, T], dim=self.f_1.shape[1:])

    def set_defect_correction(self):
        self.stepper(self.f_4, self.f_3)  # perform one step of operator on restricted finer grid approximation
        self.f_4, self.f_3 = self.f_3, self.f_4
        wp.launch(utils.add_populations, inputs=[self.f_4, self.residual, self.defect_correction, 9], dim=self.f_4.shape[1:])

    def get_error_approx(self):
        wp.launch(utils.subtract_populations, inputs=[self.f_4, self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        return self.f_3

    def perform_smoothing(self, get_residual=False):
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_2)
        #self.f_1, self.f_2 = self.f_2, self.f_1
        #wp.launch(utils.add_populations, inputs=[self.f_2, self.defect_correction, self.f_2, 9], dim=self.f_1.shape[1:])
        wp.launch(self.relax, inputs=[self.f_2, self.f_3, self.defect_correction, self.f_1, self.gamma], dim=self.f_1.shape[1:])

        if get_residual:
            self.stepper(self.f_1, self.f_3)
            return self.f_3

    def get_macroscopics(self):
        return self.stepper.get_macroscopics(self.f_1)
    
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
    i_fine = 2*i
    j_fine = 2*j

    nodes_x_fine = nodes_x_coarse * 2
    nodes_y_fine = nodes_y_coarse * 2

    for l in range(dim):
        fine[l, i_fine, j_fine, 0] = coarse[l, i, j, 0]
        fine[l, wp.mod(i_fine+1, nodes_x_fine), j_fine, 0] = 0.5 * (coarse[l, i, j, 0] + coarse[l, wp.mod(i + 1, nodes_x_coarse), j, 0])
        fine[l, i_fine, wp.mod(j_fine + 1, nodes_y_fine), 0] = 0.5 * (coarse[l, i, j, 0] + coarse[l, i, wp.mod(j + 1, nodes_y_coarse), 0])
        fine[l, wp.mod(i_fine+1, nodes_x_fine), wp.mod(j_fine+1, nodes_y_fine), 0] = 0.25 * (
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
        for i in range(self.max_levels):
            nx_level = (nodes_x - 1) // (2**i) + 1  # IMPORTANT: only works with nodes as power of two at the moment
            ny_level = (nodes_y - 1) // (2**i) + 1
            dx = length_x / float(nx_level)
            dy = length_y / float(ny_level)
            dt_level = dt * (4**i)
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

        assert self.max_levels == 2


    def work(self):
        '''self.levels[1].startup()
        macroscopics = self.levels[1].get_macroscopics()
        for i in range(min(self.timesteps, 50)):
            self.levels[1].perform_smoothing()
            macroscopics = self.levels[1].get_macroscopics()
        # now switch to fine mesh
        macroscopics_device = self.levels[1].stepper.get_macroscopics_device(self.levels[1].f_1)
        wp.launch(interpolate, inputs=[macroscopics_device, self.levels[0].macroscopics, self.levels[1].nodes_x, self.levels[1].nodes_y, 5], dim=self.levels[1].f_1.shape[1:])
        self.levels[0].startup()
        self.levels[0].init_from_macroscopics(self.levels[0].macroscopics)
        for i in range(max(self.timesteps-50, 0)):
            self.levels[0].perform_smoothing()
            macroscopics = self.levels[0].get_macroscopics()'''
        self.levels[0].startup()
        macroscopics = self.levels[0].get_macroscopics()
        #smoothing on fine grid
        for i in range(4):
            self.levels[0].perform_smoothing()
        residual = self.levels[0].perform_smoothing(get_residual=True)
        #transfer to coarse grid
        macroscopics_device = self.levels[0].stepper.get_macroscopics_device(residual)
        wp.launch(restrict, inputs=[macroscopics_device, self.levels[1].macroscopics, 5], dim = self.levels[1].f_1.shape[1:])
        self.levels[1].startup()
        self.levels[1].init_from_macroscopics(self.levels[1].macroscopics, self.levels[1].residual)
        self.levels[1].set_defect_correction()
        self.levels[1].init_from_macroscopics(self.levels[1].macroscopics, self.levels[1].f_4)
        #solve on coarse grid
        self.levels[1].startup()
        for i in range(100):
            self.levels[1].perform_smoothing()
        error_approx = self.levels[1].get_error_approx()
        macroscopics_device = self.levels[1].stepper.get_macroscopics_device(error_approx)
        #interpolate error to fine grid
        wp.launch(interpolate, inputs=[macroscopics_device, self.levels[0].macroscopics, self.levels[1].nodes_x, self.levels[1].nodes_y, 5], dim=macroscopics_device.shape[1:])
        self.levels[0].startup()
        self.levels[0].init_from_macroscopics(self.levels[0].macroscopics, self.levels[0].f_4)
        #add error to current approximation
        wp.launch(utils.add_populations, inputs=[self.levels[0].f_1, self.levels[0].f_4, self.levels[0].f_1, 9], dim=self.levels[0].f_1.shape[1:])
        for i in range(4):
            self.levels[0].perform_smoothing()
        macroscopics = self.levels[0].get_macroscopics()

        '''macroscopics_device = self.levels[1].stepper.get_macroscopics_device(self.levels[1].f_1)
        wp.launch(interpolate, inputs=[macroscopics_device, self.levels[0].macroscopics, self.levels[1].nodes_x, self.levels[1].nodes_y, 5], dim=self.levels[1].f_1.shape[1:])
        macroscopics_2 = self.levels[0].macroscopics.numpy()'''
        macroscopics = self.levels[0].get_macroscopics()
        return macroscopics