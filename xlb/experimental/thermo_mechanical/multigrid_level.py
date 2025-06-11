import numpy as np
import warp as wp
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.multigrid_stepper import MultigridStepper
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import xlb.experimental.thermo_mechanical.solid_bounceback as bc
from xlb import DefaultConfig
import math
from typing import Any
import sympy
from xlb.operator import Operator
from xlb.experimental.thermo_mechanical.multigrid_prolongation import Prolongation
from xlb.experimental.thermo_mechanical.multigrid_restriction import Restriction


class Level(Operator):
    def __init__(
        self,
        nodes_x,
        nodes_y,
        dx,
        dt,
        force_load,
        gamma,
        v1,
        v2,
        level_num,
        compute_backend,
        velocity_set,
        precision_policy,
        coarsest_level_iter=0,
    ):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        self.grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        # params needed to set up simulation params
        self.gamma = gamma
        self.dx = dx
        self.dt = dt
        self.set_params()
        # setup grids
        self.f_1 = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        self.f_2 = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        self.f_3 = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        self.f_4 = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        self.f_5 = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        self.defect_correction = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        # setup stepper
        self.stepper = MultigridStepper(self.grid, force_load, self.gamma)
        self.v1 = v1
        self.v2 = v2
        self.level_num = level_num
        self.coarsest_level_iter = coarsest_level_iter

        self.prolongation = Prolongation(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.restriction = Restriction(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

    def _construct_warp(self):
        kernel_provider = KernelProvider()
        vec = kernel_provider.vec
        read_local_population = kernel_provider.read_local_population
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global
        self.set_population_zero = kernel_provider.set_population_to_zero
        self.copy_populations = kernel_provider.copy_populations
        self.subtract_populations = kernel_provider.subtract_populations
        self.convert_populations_to_moments = kernel_provider.convert_populations_to_moments
        self.convert_moments_to_populations = kernel_provider.convert_moments_to_populations

        @wp.func
        def functional(f_previous_pre_collision: vec, f_pre_collision: vec, defect_correction: vec):
            _f_out = defect_correction
            for l in range(self.velocity_set.q):
                _f_out[l] += f_previous_pre_collision[l] - f_pre_collision[l]
            
            return _f_out

        @wp.kernel
        def kernel(
            f_1: wp.array4d(dtype=self.store_dtype), #previous pre-collision population
            f_2: wp.array4d(dtype=self.store_dtype), #new pre-collision population & output array
            defect_correction: wp.array4d(dtype=self.store_dtype),
        ):
            i, j, k = wp.tid()

            _f_previous_pre_collision = read_local_population(f_1, i, j)
            _f_pre_collision = read_local_population(f_2, i, j)
            _defect_correction = read_local_population(defect_correction, i, j)

            _f_out = functional(f_previous_pre_collision=_f_previous_pre_collision, f_pre_collision=_f_pre_collision, defect_correction=_defect_correction)

            write_population_to_global(f=f_2, f_local=_f_out, x=i, y=j)
        
        return functional, kernel
    
    def get_residual(self, f_1, f_2, f_3, defect_correction):
        self.stepper.collide(f_1, f_2)
        self.stepper.stream(f_2, f_3)
        wp.launch(self.warp_kernel, inputs=[f_1, f_3, defect_correction], dim=self.f_1.shape[1:])

    def set_params(self):
        simulation_params = SimulationParams()
        simulation_params.set_dx_dt(self.dx, self.dt)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, multigrid, return_residual=False, timestep=0):
        self.set_params()

        params = SimulationParams()
        theta = params.theta

        for i in range(self.v1):
            self.stepper(self.f_1, self.f_2, self.defect_correction)
        
        coarse = multigrid.get_next_level(self.level_num)

        if coarse is not None:

            self.get_residual(self.f_1, self.f_2, self.f_3, self.defect_correction) #f_3 now contains the residual

            self.restriction(
                fine=self.f_3, coarse=coarse.defect_correction
            )  # restrict residual to defect correction of coarser grid

            wp.launch(self.set_population_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])

            coarse(multigrid)

            wp.launch(self.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
            #print(coarse.f_1.numpy())
            self.prolongation(
                fine=self.f_1, coarse=coarse.f_1
            )  # prolongate error approx back to fine grid and add it to current solution
            wp.launch(self.copy_populations, inputs=[self.f_3, self.f_4, 9], dim=self.f_1.shape[1:])
            wp.launch(self.subtract_populations, inputs=[self.f_4, self.f_1, self.f_5, 9], dim=self.f_1.shape[1:])
            wp.launch(self.convert_populations_to_moments, inputs=[self.f_3, self.f_3], dim=self.f_1.shape[1:])
            wp.launch(self.convert_populations_to_moments, inputs=[self.f_5, self.f_5], dim=self.f_3.shape[1:])
            macroscopics = self.f_3.numpy()
            error_macroscopics = self.f_5.numpy()
            utils.plot_x_slice(array1=macroscopics[7,:,:,0], array2=error_macroscopics[7,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_f', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[6,:,:,0], array2=error_macroscopics[6,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_21', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[5,:,:,0], array2=error_macroscopics[5,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_12', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[4,:,:,0], array2=error_macroscopics[4,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_d', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[3,:,:,0], array2=error_macroscopics[3,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_s', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[2,:,:,0], array2=error_macroscopics[2,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_11', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[1,:,:,0], array2=error_macroscopics[1,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_01', label1='current', label2='error')
            utils.plot_x_slice(array1=macroscopics[0,:,:,0], array2=error_macroscopics[0,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_10', label1='current', label2='error')
        else:
            for i in range(self.coarsest_level_iter):
                self.stepper(self.f_1, self.f_2, self.defect_correction)
        
        #for i in range(self.v2):
        #    self.stepper(self.f_1, self.f_2, self.defect_correction)
        
        #for calculating WUs
        benchmark_data = BenchmarkData()
        benchmark_data.wu += (self.v1+self.v2)*0.25**self.level_num
        if coarse is None:
            benchmark_data.wu += self.coarsest_level_iter*0.25**self.level_num

        if return_residual:
            return self.stepper.get_residual_norm(self.f_1, self.f_2)
