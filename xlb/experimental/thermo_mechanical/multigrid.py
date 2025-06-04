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
        self.interpolate_bc = kernel_provider.interpolate_through_moments_with_boundaries
        self.interpolate = kernel_provider.interpolate_through_moments_no_boundaries
        self.restrict = kernel_provider.restrict_no_boundaries
        self.restrict_bc = kernel_provider.restrict_with_boundaries
        self.copy_populations = kernel_provider.copy_populations
        self.add_populations = kernel_provider.add_populations
        self.subtract_populations = kernel_provider.subtract_populations
        self.multiply_populations = kernel_provider.multiply_populations
        self.set_population_to_zero = kernel_provider.set_population_to_zero
        self.l2_norm_squared = kernel_provider.l2_norm
        self.set_zero_outside_boundary = kernel_provider.set_zero_outside_boundary
        self.convert_populations_to_moments = kernel_provider.convert_populations_to_moments
        self.convert_moments_to_populations = kernel_provider.convert_moments_to_populations
        self.check_for_nans = kernel_provider.check_for_nans

        self.boundary_conditions=None

        if self.level_num != 0:
            wp.launch(self.set_population_to_zero, inputs=[self.stepper.force, 2], dim=self.stepper.force.shape[1:])

    def get_residual_norm(self, residual):
        residual_norm_sq = wp.zeros(shape=1, dtype=self.precision_policy.compute_precision.wp_dtype)
        wp.launch(self.l2_norm_squared, inputs=[residual, residual_norm_sq], dim=residual.shape[1:])
        return math.sqrt((1 / (residual.shape[0] * residual.shape[1] * residual.shape[2])) * residual_norm_sq.numpy()[0])

    def add_boundary_conditions(self, boundary_conditions, boundary_values):
        self.boundary_conditions = boundary_conditions
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

    def get_macroscopics(self, population=None, device=False):
        self.set_params()
        if population == None:
            population = self.f_1
        if device:
            return self.stepper.get_macroscopics_device(population)
        return self.stepper.get_macroscopics_host(population)

    def set_params(self):
        simulation_params = SimulationParams()
        simulation_params.set_dx_dt(self.dx, self.dt)


    def start_v_cycle(self, return_residual=False,timestep=0):
        self.set_params()

        if self.stepper.boundary_conditions != None:
            wp.launch(self.set_zero_outside_boundary, inputs=[self.defect_correction, self.stepper.boundary_conditions], dim=self.defect_correction.shape[1:])

        # do pre-smoothing
        for i in range(self.v1):
            self.perform_smoothing()

        coarse = self.multigrid.get_next_level(self.level_num)

        '''wp.launch(self.convert_populations_to_moments, inputs=[self.f_1, self.f_1], dim=self.f_1.shape[1:])
        wp.launch(self.convert_populations_to_moments, inputs=[self.f_3, self.f_3], dim=self.f_3.shape[1:])
        macroscopics = self.f_1.numpy()
        wp.launch(self.restrict, inputs=[coarse.f_1, self.f_1], dim=coarse.f_1.shape[1:])
        wp.launch(self.interpolate, inputs=[self.f_3, coarse.f_1, coarse.nodes_x, coarse.nodes_y], dim=self.f_3.shape[1:])
        #error_macroscopics = coarse.f_1.numpy()
        error_macroscopics = self.f_3.numpy() 
        utils.plot_x_slice(array1=macroscopics[7,:,:,0], array2=error_macroscopics[7,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_f', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[6,:,:,0], array2=error_macroscopics[6,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_21', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[5,:,:,0], array2=error_macroscopics[5,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_12', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[4,:,:,0], array2=error_macroscopics[4,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_d', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[3,:,:,0], array2=error_macroscopics[3,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_s', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[2,:,:,0], array2=error_macroscopics[2,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_11', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[1,:,:,0], array2=error_macroscopics[1,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_01', label1='current', label2='error')
        utils.plot_x_slice(array1=macroscopics[0,:,:,0], array2=error_macroscopics[0,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_10', label1='current', label2='error')'''

        if coarse != None:
            # get residual
            residual = self.get_residual()
            wp.launch(self.set_zero_outside_boundary, inputs=[residual, self.boundary_conditions], dim=residual.shape[1:])
            #restrict residual to defect_corrrection on coarse grid
            wp.launch(self.restrict_bc, inputs=[coarse.defect_correction, residual, self.boundary_conditions], dim=coarse.defect_correction.shape[1:])
            wp.launch(self.set_zero_outside_boundary, inputs=[coarse.defect_correction, coarse.boundary_conditions], dim=coarse.defect_correction.shape[1:])
            # set intial guess of coarse mesh to residual
            wp.launch(self.set_population_to_zero, inputs=[coarse.f_1, 9], dim=coarse.f_1.shape[1:])
            wp.launch(self.set_population_to_zero, inputs=[coarse.f_4, 9], dim=coarse.f_1.shape[1:])

            # scale defect correction?
            wp.launch(self.multiply_populations, inputs=[coarse.defect_correction, 4., 9], dim=coarse.defect_correction.shape[1:])
            # start v_cycle on coarse grid
            coarse.start_v_cycle(timestep=timestep)
            # get approximation of error
            error_approx = coarse.f_1
            # interpolate error approx to fine grid
            wp.launch(self.interpolate_bc, inputs=[self.f_3, error_approx, coarse.nodes_x, coarse.nodes_y, coarse.boundary_conditions], dim=self.f_3.shape[1:])
            if self.boundary_conditions != None:
                wp.launch(self.set_zero_outside_boundary, inputs=[self.f_3, self.boundary_conditions], dim=self.f_3.shape[1:])

            if (self.level_num == 0 and True):
                macroscopics = self.get_macroscopics(self.f_1)
                error_macroscopics = self.get_macroscopics(self.f_3)
                wp.launch(self.convert_populations_to_moments, inputs=[residual, residual], dim=residual.shape[1:])
                m_res = residual.numpy() 
                #convert current and error approx to moments
                wp.launch(self.convert_populations_to_moments, inputs=[self.f_1, self.f_1], dim=self.f_1.shape[1:])
                wp.launch(self.convert_populations_to_moments, inputs=[self.f_3, self.f_3], dim=self.f_3.shape[1:])
                wp.launch(self.convert_populations_to_moments, inputs=[error_approx, coarse.f_2], dim=error_approx.shape[1:])
                macroscopics = self.f_1.numpy()
                error_macroscopics = self.f_3.numpy()
                utils.plot_x_slice(array1=macroscopics[7,:,:,0], array2=error_macroscopics[7,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_f', label1='current', label2='error')
                utils.plot_x_slice(array1=coarse.f_2.numpy()[7,:,:,0], array2=error_macroscopics[7,:,:,0], dx1=coarse.dx, dx2=self.dx, timestep=timestep, name='m_f_2', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[7,:,:,0], array2=error_macroscopics[7,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_f', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[6,:,:,0], array2=error_macroscopics[6,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_21', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[5,:,:,0], array2=error_macroscopics[5,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_12', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[4,:,:,0], array2=error_macroscopics[4,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_d', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[3,:,:,0], array2=error_macroscopics[3,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_s', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[2,:,:,0], array2=error_macroscopics[2,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_11', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[1,:,:,0], array2=error_macroscopics[1,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_01', label1='current', label2='error')
                utils.plot_x_slice(array1=macroscopics[0,:,:,0], array2=error_macroscopics[0,:,:,0], dx1=self.dx, dx2=self.dx, timestep=timestep, name='m_10', label1='current', label2='error')

                wp.launch(self.convert_moments_to_populations, inputs=[self.f_1, self.f_1], dim=self.f_1.shape[1:])
                wp.launch(self.convert_moments_to_populations, inputs=[self.f_3, self.f_3], dim=self.f_3.shape[1:])

            # add error_approx to current estimate
            #wp.launch(self.add_populations, inputs=[self.f_1, self.f_3, self.f_1, 9], dim=self.f_1.shape[1:])
        
        #copy over to f_4 for boundary conditions
        wp.launch(self.copy_populations, inputs=[self.f_1, self.f_4, 9], dim=self.f_1.shape[1:])

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
        output_images=False,
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
                    boundary_conditions_level, boundary_values_level = bc.init_bc_from_lambda(potential_sympy=potential, grid=level.grid, dx=dx, velocity_set=velocity_set, manufactured_displacement=displacement, indicator=indicator, x=x, y=y, precision_policy=precision_policy)
                    #print(boundary_values.numpy())
                    #level.boundary_conditions = boundary_conditions_level
                    level.add_boundary_conditions(boundary_conditions_level, boundary_values_level)

            self.levels.append(level)

    def get_next_level(self, level_num):
        if level_num + 1 < self.max_levels:
            return self.levels[level_num + 1]
        else:
            return None

    def get_finest_level(self):
        return self.levels[0]
