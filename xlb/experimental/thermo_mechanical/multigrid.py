import numpy as np
import warp as wp
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.experimental.thermo_mechanical.solid_utils as utils
from  xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
import math
from typing import Any


class Level:
    def __init__(self, nodes_x, nodes_y, dx, dt, force_load, gamma, v1, v2, level_num, multigrid, compute_backend, velocity_set, precision_policy):
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
        self.multigrid = multigrid
        # setup grids
        self.f_1 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_2 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_3 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_4 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.macroscopics = self.grid.create_field(cardinality=9, dtype=precision_policy.store_precision) 
        self.residual = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.defect_correction = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        # setup stepper
        self.stepper = SolidsStepper(self.grid, force_load)
        self.v1 = v1
        self.v2 = v2
        self.level_num = level_num

        @wp.kernel
        def relaxation(f_after_stream: Any, f_previous: Any, defect_correction: Any, f_destination: Any, gamma: Any):
            i, j, k = wp.tid()
            for l in range(velocity_set.q):
                f_destination[l, i, j, 0] = (
                    gamma * (f_after_stream[l, i, j, 0] - defect_correction[l, i, j, 0]) + (1.0 - gamma) * f_previous[l, i, j, 0]
                )
        self.relax = relaxation


    def startup(self):
        solid_simulation = SimulationParams()
        E = solid_simulation.E_unscaled
        nu = solid_simulation.nu_unscaled
        kappa = solid_simulation.kappa
        theta = solid_simulation.theta
        solid_simulation.set_parameters(E=E, nu=nu, dx=self.dx, dt=self.dt, L=self.dx, T=self.dt, kappa=kappa, theta=theta)
    
  
    def perform_smoothing(self):
        #for statistics
        benchmark_data = BenchmarkData()
        benchmark_data.wu += (0.25**self.level_num)
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_2)
        wp.launch(self.relax, inputs=[self.f_2, self.f_3, self.defect_correction, self.f_1, self.gamma], dim=self.f_1.shape[1:])


    def get_residual(self):
        self.startup()
        wp.launch(utils.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        wp.launch(utils.add_populations, inputs=[self.f_1, self.defect_correction, self.residual, 9], dim=self.f_1.shape[1:])
        #wp.launch(utils.subtract_populations, inputs=[self.defect_correction, self.f_1, self.residual,9], dim=self.defect_correction.shape[1:])
        self.stepper(self.f_3, self.f_2)
        #rules for operator: A(f) = current - previous
        # --> residual = defect - A(f) = defect + previous - current
        wp.launch(utils.subtract_populations, inputs=[self.residual, self.f_2, self.residual, 9], dim=self.residual.shape[1:])
        #wp.launch(utils.add_populations, inputs=[self.residual, self.f_2, self.residual, 9], dim=self.residual.shape[1:])
        return self.residual



    
    def get_macroscopics(self):
        self.startup()
        return self.stepper.get_macroscopics_host(self.f_1)
    

    def start_v_cycle(self):
        #do pre-smoothing
        for i in range(self.v1):
            self.perform_smoothing()
        
        coarse = self.multigrid.get_next_level(self.level_num)
        if (coarse != None):
            #get residual
            residual = self.get_residual()
            #restrict residual to defect_corrrection on coarse grid
            wp.launch(restrict, inputs=[coarse.defect_correction, residual, self.nodes_x, self.nodes_y, 9], dim=coarse.defect_correction.shape[1:])
            #set intial guess of coarse mesh to zero
            wp.launch(utils.set_population_to_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])
            #scale defect correction?
            wp.launch(utils.multiply_populations, inputs=[coarse.defect_correction, 4., 9], dim=coarse.defect_correction.shape[1:])
            #start v_cycle on coarse grid
            coarse.start_v_cycle()
            #get approximation of error
            error_approx = coarse.f_1
            #print("Coarse: {}".format(np.max(error_approx.numpy())))
            #interpolate error approx to fine grid
            wp.launch(interpolate, inputs=[self.f_3, error_approx, 9], dim=self.f_3.shape[1:])
            #scale correction?
            #wp.launch(utils.multiply_populations, inputs=[self.f_3, 0.25, 9], dim=self.f_3.shape[1:])
            #add error_approx to current estimate
            #print("Fine: {}".format(np.max(self.f_3.numpy())))
            if (self.level_num == 1):
                print("Defect Correction: {}".format(np.max(self.defect_correction.numpy())))
                print("Error approx: {}".format(np.max(self.f_3.numpy())))
                print(self.f_3.numpy()[1,:,:,0])
                print(self.f_1.numpy()[1,:,:,0])
            #wp.launch(utils.add_populations, inputs=[self.f_1, self.f_3, self.f_1, 9], dim=self.f_1.shape[1:])

        #do post_smoothing
        for i in range(self.v2):
            self.perform_smoothing

        residual_host = self.get_residual().numpy()
        return np.max(residual_host)





   
@wp.kernel
def interpolate(fine: Any, coarse: Any, dim: Any):
    i,j,k = wp.tid()

    if (wp.mod(i, 2) == 0) and (wp.mod(j, 2) == 0) or True:
        coarse_i = i/2
        coarse_j = j/2 #check if really rounds down!
        #printf("%d, %d, %d, %d\n", i, coarse_i, j, coarse_j)

        for l in range(dim):
            fine[l,i,j,0] = coarse[l,coarse_i,coarse_j,0]
    else:
        for l in range(dim):
            fine[l,i,j,0] = 0.
   

@wp.kernel
def restrict(coarse: Any, fine: Any, fine_nodes_x: Any, fine_nodes_y: Any, dim: Any):
   i,j,k = wp.tid()

   for l in range(dim):
        val =  0.
        val += fine[l, 2*i, 2*j, 0]
        #val += fine[l, wp.mod(2*i+1, fine_nodes_x), 2*j, 0]
        #val += fine[l, 2*i, wp.mod(2*j+1, fine_nodes_y), 0]
        #val += fine[l, wp.mod(2*i+1, fine_nodes_x), wp.mod(2*j+1, fine_nodes_y), 0]
        val += fine[l, 2*i+1, 2*j, 0]
        val += fine[l, 2*i, 2*j+1, 0]
        val += fine[l, 2*i+1, 2*j+1, 0]
        coarse[l, i, j, 0] = 0.25*val







class MultigridSolver:
    """
    A class implementing a multigrid iterative solver for elliptic PDEs.
    """

    def __init__(self, nodes_x, nodes_y, length_x, length_y, dt, E, nu, force_load, gamma, v1, v2, max_levels=None):
        compute_backend = ComputeBackend.WARP
        precision_policy = PrecisionPolicy.FP32FP32
        velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)
        xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)

        solid_simulation = SimulationParams()
        solid_simulation.set_parameters(
            E=E, nu=nu, dx=1, dt=1, L=1, T=1, kappa=1, theta=1.0 / 3.0
        )  # just placeholder, so E and nu get past to all levels

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
            dt_level = dt*(4**i)
            assert math.isclose(dx, dy)
            level = Level(
                nodes_x=nx_level,
                nodes_y=ny_level,
                dx=dx,
                dt=dt_level,
                force_load=force_load,
                gamma=gamma,
                v1=v1,
                v2=v2,
                level_num=i,
                multigrid=self,
                compute_backend=compute_backend,
                velocity_set=velocity_set,
                precision_policy=precision_policy,
            )
            if (i != 0):
                wp.launch(utils.set_population_to_zero, inputs=[level.stepper.force, 2], dim=level.stepper.force.shape[1:])

            self.levels.append(level)


    def get_next_level(self, level_num):
        if level_num + 1 < self.max_levels:
            return self.levels[level_num + 1]
        else:
            return None

    def get_finest_level(self):
        return self.levels[0]
