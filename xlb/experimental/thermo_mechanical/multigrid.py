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
        wp.config.mode = "debug"
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
        self.macroscopics = self.grid.create_field(cardinality=9, dtype=precision_policy.store_precision)  # 5 macroscopic variables
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
                #f_destination[l, i, j, 0] = f_after_stream[l, i, j, 0] + defect_correction[l, i, j, 0]

        self.relax = relaxation

    def set_defect_correction(self):
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_4, self.f_2, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_4, self.f_3)  # perform one step of operator on restricted finer grid approximation  
        wp.launch(utils.subtract_populations, inputs=[self.f_2, self.f_3, self.f_4, 9], dim=self.f_4.shape[1:])
        wp.launch(utils.multiply_populations, inputs=[self.residual, 4., 9], dim=self.f_4.shape[1:])
        wp.launch(utils.subtract_populations, inputs=[self.f_4, self.residual, self.defect_correction, 9], dim=self.f_4.shape[1:])
        wp.launch(utils.copy_populations, inputs=[self.f_2, self.f_4, 9], dim=self.f_1.shape[1:])

    def startup(self):
        solid_simulation = SimulationParams()
        E = solid_simulation.E_unscaled
        nu = solid_simulation.nu_unscaled
        kappa = solid_simulation.kappa
        theta = solid_simulation.theta
        solid_simulation.set_parameters(E=E, nu=nu, dx=self.dx, dt=self.dt, L=self.dx, T=self.dt, kappa=kappa, theta=theta)
    
  
    def perform_smoothing(self, get_residual=False):
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_2)
        #self.f_1, self.f_2 = self.f_2, self.f_1
        # wp.launch(utils.add_populations, inputs=[self.f_2, self.defect_correction, self.f_2, 9], dim=self.f_1.shape[1:])
        wp.launch(self.relax, inputs=[self.f_2, self.f_3, self.defect_correction, self.f_1, self.gamma], dim=self.f_1.shape[1:])

        if get_residual:
            wp.launch(utils.subtract_populations, inputs=[self.f_3, self.f_1, self.residual, self.velocity_set.q], dim=self.f_1.shape[1:])
            return self.residual
    
    def get_error_approx(self):
        wp.launch(utils.subtract_populations, inputs=[self.f_4, self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        return self.f_3


    def get_macroscopics(self):
        self.startup()
        return self.stepper.get_macroscopics_host(self.f_1)
    
   
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
            dt_level = dt 
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
        fine = self.levels[0] 
        coarse = self.levels[1]

        for i in range(2):
            fine.perform_smoothing(get_residual=False)
        residual = fine.perform_smoothing(get_residual=True)

        #wp.launch(utils.set_population_to_zero, inputs=[coarse.stepper.force, 2], dim=coarse.stepper.force.shape[1:])
        #wp.launch(utils.set_population_to_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])
        wp.launch(restrict, inputs=[coarse.f_1, fine.f_1, 9], dim=coarse.f_1.shape[1:])
        wp.launch(restrict, inputs=[coarse.residual, residual, 9], dim=coarse.defect_correction.shape[1:])
        wp.launch(restrict, inputs=[coarse.f_4, fine.f_1, 9], dim=coarse.f_4.shape[1:])
        coarse.set_defect_correction()

        
        for i in range(4):
            residual = coarse.perform_smoothing(get_residual=True)
#           print(np.linalg.norm(residual.numpy()))

        
        error_approx = coarse.get_error_approx()
        wp.launch(interpolate, inputs=[error_approx, fine.f_4, coarse.nodes_x, coarse.nodes_y, 9], dim=coarse.f_4.shape[1:])
        wp.launch(utils.multiply_populations, inputs=[fine.f_4, 0.25, 9], dim=fine.f_4.shape[1:])

        wp.launch(utils.add_populations, inputs=[fine.f_1, fine.f_4, fine.f_1, 9], dim=fine.f_1.shape[1:])

        for i in range(2):
            fine.perform_smoothing(get_residual=False)

        return fine.get_macroscopics()

        

