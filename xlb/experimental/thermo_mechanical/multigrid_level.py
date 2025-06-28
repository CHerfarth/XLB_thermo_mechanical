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
        error_correction_iterations=1, #by default do V-cycle
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
        self.boundary_conditions = None
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
        self.defect_correction = self.grid.create_field(
            cardinality=velocity_set.q, dtype=precision_policy.store_precision
        )
        # setup stepper
        self.stepper = MultigridStepper(self.grid, force_load, self.gamma)
        self.v1 = v1
        self.v2 = v2
        self.error_correction_iterations = error_correction_iterations
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
        self.set_zero_outside_boundary = kernel_provider.set_zero_outside_boundary

        @wp.func
        def functional(f_previous_pre_collision: vec, f_pre_collision: vec, defect_correction: vec):
            _f_out = defect_correction
            for l in range(self.velocity_set.q):
                _f_out[l] += f_previous_pre_collision[l] - f_pre_collision[l]

            return _f_out

        @wp.kernel
        def kernel(
            f_1: wp.array4d(dtype=self.store_dtype),  # previous pre-collision population & output array
            f_2: wp.array4d(dtype=self.store_dtype),  # new pre-collision population 
            defect_correction: wp.array4d(dtype=self.store_dtype),
        ):
            i, j, k = wp.tid()

            _f_previous_pre_collision = read_local_population(f_1, i, j)
            _f_pre_collision = read_local_population(f_2, i, j)
            _defect_correction = read_local_population(defect_correction, i, j)

            _f_out = functional(
                f_previous_pre_collision=_f_previous_pre_collision,
                f_pre_collision=_f_pre_collision,
                defect_correction=_defect_correction,
            )

            write_population_to_global(f=f_1, f_local=_f_out, x=i, y=j)

        return functional, kernel

    def get_residual(self, f_1, f_2, f_3, defect_correction):
        wp.launch(self.copy_populations, inputs=[f_1, f_3, 9], dim=f_1.shape[1:])
        self.stepper(f_1, f_2, self.f_4, defect_correction, gamma=1., defect_factor=0.)
        wp.launch(self.warp_kernel, inputs=[f_3, f_1, defect_correction], dim=f_1.shape[1:])

    def set_params(self):
        simulation_params = SimulationParams()
        simulation_params.set_dx_dt(self.dx, self.dt)

    def add_boundary_conditions(self, boundary_conditions, boundary_values):
        self.boundary_conditions = boundary_conditions
        self.stepper.add_boundary_conditions(boundary_conditions, boundary_values)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, multigrid, return_residual=False, timestep=0):
        #set new dx, dt
        self.set_params()
        params = SimulationParams()
        theta = params.theta

        #pre-smooth
        for i in range(self.v1):
            self.stepper(self.f_1, self.f_2, self.f_4, self.defect_correction)

        coarse = multigrid.get_next_level(self.level_num)

        #start MG iteration on coarse grid
        if coarse is not None:
            self.get_residual(
                self.f_1, self.f_2, self.f_3, self.defect_correction
            )  # f_3 now contains the residual

            
            if self.boundary_conditions is None:
                self.restriction(
                    fine=self.f_3, coarse=coarse.defect_correction
                )  # restrict residual to defect correction of coarser grid
            else:
                self.restriction(fine=self.f_3, coarse=coarse.defect_correction, fine_boundary_array=self.boundary_conditions)

            wp.launch(self.set_population_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])

            #recursive calls to MG on coarse grid
            for i in range(self.error_correction_iterations):
                coarse(multigrid)

            if self.boundary_conditions is None: 
                self.prolongation(
                    fine=self.f_1, coarse=coarse.f_1
                )  # prolongate error approx back to fine grid and add it to current solution
            else:
                self.prolongation(fine=self.f_1, coarse=coarse.f_1, coarse_boundary_array=coarse.boundary_conditions)
            self.set_params()
        # or solve directly
        else:
            for i in range(self.coarsest_level_iter):
                self.stepper(self.f_1, self.f_2, self.f_4, self.defect_correction)

        #post-smooth
        for i in range(self.v2):
            if i == 0:
                self.stepper(self.f_1, self.f_2, self.f_4, self.defect_correction, f_3_uninitialized=True)
            else:
                self.stepper(self.f_1, self.f_2, self.f_4, self.defect_correction)


        # for calculating WUs
        benchmark_data = BenchmarkData()
        benchmark_data.wu += (self.v1 + self.v2) * 0.25**self.level_num
        if coarse is None:
            benchmark_data.wu += self.coarsest_level_iter * 0.25**self.level_num

        if return_residual:
            return self.stepper.get_residual_norm(self.f_1, self.f_2)
