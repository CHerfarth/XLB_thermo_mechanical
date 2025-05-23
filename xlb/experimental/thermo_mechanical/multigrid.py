import numpy as np
import warp as wp
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.benchmark_data import BenchmarkData
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import xlb.experimental.thermo_mechanical.solid_bounceback as bc
from xlb import DefaultConfig
import math
from typing import Any
import sympy


class Level:
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
        multigrid,
        compute_backend,
        velocity_set,
        precision_policy,
        coarsest_level_iter=0,
    ):
        wp.config.mode = "debug"
        self.grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.velocity_set = velocity_set
        self.precision_policy = precision_policy
        c = self.velocity_set.c_float
        # params needed to set up simulation params
        self.gamma = gamma
        self.dx = dx
        self.dt = dt
        self.multigrid = multigrid
        # setup grids
        self.f_1 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_2 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_3 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.f_4 = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.macroscopics = self.grid.create_field(cardinality=9, dtype=precision_policy.store_precision)
        self.defect_correction = self.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
        self.residual_norm_sq = None
        # setup stepper
        self.stepper = SolidsStepper(self.grid, force_load)
        self.v1 = v1
        self.v2 = v2
        self.level_num = level_num
        self.coarsest_level_iter = coarsest_level_iter

        # get all necessary kernels
        kernel_provider = KernelProvider()
        self.relax = kernel_provider.relaxation
        self.interpolate = kernel_provider.interpolate_through_moments
        # self.interpolate = kernel_provider.interpolate
        # self.restrict = kernel_provider.restrict
        self.restrict = kernel_provider.restrict_through_moments
        self.copy_populations = kernel_provider.copy_populations
        self.add_populations = kernel_provider.add_populations
        self.subtract_populations = kernel_provider.subtract_populations
        self.multiply_populations = kernel_provider.multiply_populations
        self.set_population_to_zero = kernel_provider.set_population_to_zero
        self.l2_norm_squared = kernel_provider.l2_norm
        self.set_zero_outside_boundary = kernel_provider.set_zero_outside_boundary

        if self.level_num != 0:
            wp.launch(self.set_population_to_zero, inputs=[self.stepper.force, 2], dim=self.stepper.force.shape[1:])

    def get_residual_norm(self, residual):
        residual_norm_sq = wp.zeros(shape=1, dtype=self.precision_policy.compute_precision.wp_dtype)
        wp.launch(self.l2_norm_squared, inputs=[residual, residual_norm_sq], dim=residual.shape[1:])
        return math.sqrt((1 / (residual.shape[0] * residual.shape[1] * residual.shape[2])) * residual_norm_sq.numpy()[0])

    def add_boundary_conditions(self, boundary_conditions, boundary_values):
        self.stepper.add_boundary_conditions(boundary_conditions, boundary_values)

    def perform_smoothing(self):
        # for statistics
        benchmark_data = BenchmarkData()
        benchmark_data.wu += 0.25**self.level_num
        wp.launch(self.copy_populations, inputs=[self.f_1, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_1, self.f_4)
        wp.launch(self.relax, inputs=[self.f_4, self.f_3, self.defect_correction, self.f_4, self.gamma, self.velocity_set.q], dim=self.f_1.shape[1:])
        self.f_1, self.f_4 = self.f_4, self.f_1

    def get_residual(self):
        wp.launch(self.copy_populations, inputs=[self.f_1, self.f_2, 9], dim=self.f_1.shape[1:])
        wp.launch(self.copy_populations, inputs=[self.f_4, self.f_3, 9], dim=self.f_1.shape[1:])
        self.stepper(self.f_2, self.f_3)
        # rules for operator: A(f) = current - previous
        # --> residual = defect - A(f) = defect + previous - current
        wp.launch(self.add_populations, inputs=[self.f_1, self.defect_correction, self.f_2, 9], dim=self.f_1.shape[1:])
        wp.launch(self.subtract_populations, inputs=[self.f_2, self.f_3, self.f_2, 9], dim=self.f_2.shape[1:])
        # if simulating with boundary conditions, set residual to 0 outside potential
        if self.stepper.boundary_conditions != None:
            wp.launch(self.set_zero_outside_boundary, inputs=[self.f_2, self.stepper.boundary_conditions], dim=self.f_2.shape[1:])
        return self.f_2

    def get_macroscopics(self):
        return self.stepper.get_macroscopics_host(self.f_1)

    def start_v_cycle(self, return_residual=False):
        # do pre-smoothing
        for i in range(self.v1):
            self.perform_smoothing()

        coarse = self.multigrid.get_next_level(self.level_num)
        if coarse != None:
            # get residual
            residual = self.get_residual()
            # restrict residual to defect_corrrection on coarse grid
            wp.launch(self.restrict, inputs=[coarse.defect_correction, residual], dim=coarse.defect_correction.shape[1:])
            # set intial guess of coarse mesh to residual
            wp.launch(self.set_population_to_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])
            # scale defect correction?
            wp.launch(self.multiply_populations, inputs=[coarse.defect_correction, 4.0, 9], dim=coarse.defect_correction.shape[1:])
            # start v_cycle on coarse grid
            coarse.start_v_cycle()
            # get approximation of error
            error_approx = coarse.f_1
            # interpolate error approx to fine grid
            wp.launch(self.interpolate, inputs=[self.f_3, error_approx, coarse.nodes_x, coarse.nodes_y], dim=self.f_3.shape[1:])
            # if boundary conditions enabled, dont update solution based on ghost node values
            if self.stepper.boundary_conditions != None:
                wp.launch(self.set_zero_outside_boundary, inputs=[self.f_3, self.stepper.boundary_conditions], dim=self.f_3.shape[1:])
            # add error_approx to current estimate
            wp.launch(self.add_populations, inputs=[self.f_1, self.f_3, self.f_1, 9], dim=self.f_1.shape[1:])

        # do post_smoothing
        for i in range(self.v2):
            self.perform_smoothing()

        if coarse == None:
            for i in range(self.coarsest_level_iter):
                self.perform_smoothing()

        if return_residual:
            return self.get_residual_norm(self.get_residual())
        else:
            return None


class MultigridSolver:
    """
    A class implementing a multigrid iterative solver for elliptic PDEs.
    """

    def __init__(
        self,
        nodes_x,
        nodes_y,
        length_x,
        length_y,
        dt,
        force_load,
        gamma,
        v1,
        v2,
        max_levels=None,
        coarsest_level_iter=0,
        boundary_conditions=None,
        boundary_values=None,
        potential=None,
    ):
        precision_policy = DefaultConfig.default_precision_policy
        compute_backend = DefaultConfig.default_backend
        velocity_set = DefaultConfig.velocity_set
        # TODO: boundary conditions

        # Determine maximum possible levels
        self.max_possible_levels = min(int(np.log2(nodes_x)), int(np.log2(nodes_y)))  # + 1

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
                v1=v1,
                v2=v2,
                level_num=i,
                multigrid=self,
                compute_backend=compute_backend,
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                coarsest_level_iter=coarsest_level_iter,
            )
            if boundary_conditions != None:
                if i == 0:
                    level.add_boundary_conditions(boundary_conditions, boundary_values)
                else:
                    # create zero displacement boundary for coarser meshes
                    x, y = sympy.symbols("x y")
                    displacement = [0 * x + 0 * y, 0 * x + 0 * y]
                    indicator = lambda x, y: -1
                    boundary_conditions_level, boundary_values_level = bc.init_bc_from_lambda(
                        potential, level.grid, dx, velocity_set, displacement, indicator, x, y, precision_policy
                    )
                    level.add_boundary_conditions(boundary_conditions_level, boundary_values)
            self.levels.append(level)

    def get_next_level(self, level_num):
        if level_num + 1 < self.max_levels:
            return self.levels[level_num + 1]
        else:
            return None

    def get_finest_level(self):
        return self.levels[0]
