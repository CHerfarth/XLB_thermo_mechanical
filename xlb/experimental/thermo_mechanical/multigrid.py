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
        self.velocity_set = velocity_set
        self.f_1 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_2 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_3 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.residual = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.defect_correction = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.stepper = SolidsStepper(self.grid, force_load)
        self.gamma = gamma
        self.dx = dx
        self.dt = dt

        @wp.kernel
        def relaxation(f_after_stream: Any, f_previous: Any, defect_correction: Any,f_destination: Any,  gamma: wp.float32):
            i,j,k = wp.tid()
            for l in range(velocity_set.q):
                f_destination[l,i,j,0] = gamma*(f_after_stream[l,i,j,0] + defect_correction[l,i,j,0]) + (1. - gamma)*f_previous[l,i,j,0] 
        
        self.relax = relaxation

    def startup(self):
        solid_simulation = SimulationParams()
        E = solid_simulation.E
        nu = solid_simulation.nu
        kappa = solid_simulation.kappa
        theta = solid_simulation.theta
        solid_simulation.set_parameters(E=E, nu=nu, dx=self.dx, dt=self.dt, L=self.dx, T=self.dt, kappa=kappa, theta=theta)



    def set_defect_correction(self, defect_correction_grid):
        self.defect_correction = defect_correction_grid

    def perform_smoothing(self, get_residual = False):
        #wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_2)
        self.f_1, self.f_2 = self.f_2, self.f_1
        #wp.launch(self.relax, inputs=[self.f_2, self.f_3, self.defect_correction, self.f_1, self.gamma], dim=self.f_1.shape[1:])

        #if get_residual:
            #wp.launch(utils.get_residual, inputs=[self.f_3, self.f_1, self.residual, self.velocity_set.q])

    def get_macroscopics(self):
        return self.stepper.get_macroscopics(self.f_1)


@wp.kernel
def interpolate(coarse: Any, fine: Any, nodes_x_coarse: wp.int32, nodes_y_coarse: wp.int32, dim: wp.int8):
    i, j, k = wp.tid()

    nodes_x_fine = nodes_x_coarse * 2
    nodes_y_fine = nodes_y_coarse * 2

    for l in range(dim):
        fine[l, 2*i, 2*j, 0] = coarse[l, i, j, 0]
        fine[l, wp.mod(2*i+1, nodes_x_fine), 2*j, 0] = 0.5*(coarse[l, i, j, 0] + coarse[l, wp.mod(i+1, nodes_x_coarse), j, 0])
        fine[l, i, wp.mod(2*j+1, nodes_y_fine), 0] = 0.5*(coarse[l, i, j, 0] + coarse[l, i, wp.mod(j+1, nodes_y_coarse), 0])
        fine[l, wp.mod(2*i+1, nodes_x_fine), wp.mod(2*j+1, nodes_y_fine), 0] = 0.25*(coarse[l,i,j,0] + coarse[l,wp.mod(i+1, nodes_x_coarse), j, 0] + coarse[l,i,wp.mod(j+1,nodes_y_coarse), 0] + coarse[l, wp.mod(i+1, nodes_x_coarse), wp.mod(j+1, nodes_y_coarse), 0])

@wp.kernel
def restrict(coarse: Any, fine: Any, dim: wp.int8):
    i, j, k = wp.tid()
    for l in range(dim):
        coarse[l, i, j, 0] = fine[l, 2*i, 2*j, 0]
        



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
        solid_simulation.set_parameters(E=E, nu=nu, dx=1, dt=1, L=1, T=1, kappa=1, theta=1.0 / 3.0) #just placeholder, so E and nu get past to all levels
        self.timesteps = timesteps

        #TODO: boundary conditions


        # Determine maximum possible levels
        self.max_possible_levels = min(int(np.log2(nodes_x-1)), int(np.log2(nodes_y-1))) + 1
        
        if max_levels is None:
            self.max_levels = self.max_possible_levels
        else:
            self.max_levels = min(max_levels, self.max_possible_levels)

        #setup levels
        self.levels = list()
        for i in range(self.max_levels):
            nx_level = (nodes_x - 1) // (2 ** i) + 1 #IMPORTANT: only works with nodes as power of two at the moment
            ny_level = (nodes_y - 1) // (2 ** i) + 1
            dx = length_x / float(nx_level)
            dy = length_y / float(ny_level)
            assert math.isclose(dx, dy)
            level  = Level(nx_level, ny_level, dx, dt, force_load, gamma, compute_backend, velocity_set, precision_policy)
            self.levels.append(level)

        assert(self.max_levels == 2)
    
    def work(self):
        self.levels[0].startup()
        for i in range(self.timesteps):
            self.levels[0].perform_smoothing()
        macroscopics = self.levels[0].get_macroscopics()
        return macroscopics
            