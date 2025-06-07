import numpy as np
import warp as wp
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.solid_relaxed_stepper import SolidsRelaxedStepper
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
        self.defect_correction = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        # setup stepper
        self.stepper = SolidsRelaxedStepper(self.grid, force_load, self.gamma)
        self.v1 = v1
        self.v2 = v2
        self.level_num = level_num
        self.coarsest_level_iter = coarsest_level_iter

        self.prolongation = Prolongation(velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend)
        self.restriction = Restriction(velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend)

    def _construct_warp(self):
        kernel_provider = KernelProvider()
        vec = kernel_provider.vec
        read_local_population = kernel_provider.read_local_population
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global

        @wp.kernel
        def kernel(f_1: wp.array4d(dtype=self.store_dtype), f_2: wp.array4d(dtype=self.store_dtype), defect_correction: wp.array4d(dtype=self.store_dtype),
        force: wp.array4d(dtype=self.store_dtype),
        omega: vec,
        theta: self.compute_dtype,
        gamma: self.compute_dtype,
        ):
            i,j,k=wp.tid()
            _f_post_collision = read_local_population(f_1, i, j)
            _defect = read_local_population(defect_correction, i, j)
            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            _f_new_post_collision = self.stepper.warp_functional(i=i, j=j, k=k, f_1=f_1, defect_vec=_defect, omega=omega, force_x=force_x, force_y=force_y, theta=theta, gamma=gamma)
            # rules for operator: A(f) = current - previous
            # --> residual = defect - A(f) = defect + previous - current
            _res = vec()
            for l in range(self.velocity_set.q):
                _res[l] = self.compute_dtype(0)*_defect[l] + _f_post_collision[l] - _f_new_post_collision[l]

            write_population_to_global(f_2, _res, i, j)

        return None, kernel

    def set_params(self):
        simulation_params = SimulationParams()
        simulation_params.set_dx_dt(self.dx, self.dt)
        
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, multigrid):
        params = SimulationParams()
        theta = params.theta

        for i in range(self.v1):
            self.stepper(self.f_1, self.f_2, self.defect_correction)
            self.f_1, self.f_2 = self.f_2, self.f_1
        

        coarse = multigrid.get_next_level(self.level_num)


        if coarse != None:
            wp.launch(self.warp_kernel, inputs=[self.f_1, self.f_2, self.defect_correction, self.stepper.force, self.stepper.omega, theta, self.gamma], dim=self.f_2.shape[1:]) #f_2 now contains the residual

            self.restriction(fine=self.f_2, coarse=coarse.defect_correction) #restrict residual to defect correction of coarser grid

            coarse(multigrid)

            self.prolongation(fine=self.f_1, coarse=coarse.f_1) #prolongate error approx back to fine grid and add it to current solution
        
        for i in range(self.v2):
            self.stepper(self.f_1, self.f_2, self.defect_correction)
            self.f_1, self.f_2 = self.f_2, self.f_1
        



