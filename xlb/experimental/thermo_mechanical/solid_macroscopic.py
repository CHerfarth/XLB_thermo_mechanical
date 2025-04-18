from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.operator import Operator
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams


class SolidMacroscopics(Operator):
    def __init__(self, grid, force, omega, boundaries=None, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        self.force = force
        self.omega = omega
        self.boundaries = boundaries
        self.bared_moments = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        # Mapping for macroscopics:
        # 0: dis_x
        # 1: dis_y
        # 2: s_xx
        # 3: s_yy
        # 4: s_xy
        # 5: dx_sxx
        # 6: dy_syy
        # 7: dy_sxy
        #  8: dx_sxy 
        self.macroscopics = grid.create_field(cardinality=9, dtype=self.precision_policy.store_precision)
        if self.boundaries == None:
            self.boundaries = grid.create_field(cardinality=10, dtype=precision_policy.store_precision, fill_value=1)

        macro_vec = wp.vec(
            9, dtype=self.precision_policy.compute_precision.wp_dtype
        )  # this is the default precision policy; it can be changed by calling set_precision_policy()


        @wp.kernel
        def update(f: Any, bared_moments: Any, force: Any, omega: Any, theta: Any):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            # calculate moments
            f_local = utils.read_local_population(f, i, j)
            m = utils.calc_moments(f_local)

            # apply half-forcing and get displacement
            m[0] += 0.5 * force[0, i, j, 0]
            m[1] += 0.5 * force[1, i, j, 0]

            m_eq = utils.calc_equilibrium(m, theta)  # do something with this?
            for l in range(m._length_):
                m[l] = 0.5 * omega[l] * m_eq[l] + (1.0 - 0.5 * omega[l]) * m[l]

            utils.write_population_to_global(bared_moments, m, i, j)

        @wp.kernel
        def calc_macroscopics(macroscopics: Any, f: Any, bared_moments: Any, force: Any, L: Any, T: Any, theta: Any):
            i, j, k = wp.tid()
            f_local = utils.read_local_population(f, i, j) 
            m_local = utils.calc_moments(f_local)
            bared_m_local = utils.read_local_population(bared_moments, i, j)
            tau_t = 0.5
            dev_factor = 2./(1.+2.*tau_t)

            # calculate macroscopics
            dis_x = bared_m_local[0]
            dis_y = bared_m_local[1]
        
            bared_m_s = bared_m_local[3]
            bared_m_d = bared_m_local[4]
            bared_m_11 = bared_m_local[2]
            s_xx = -0.5 * (bared_m_s + bared_m_d)
            s_yy = -0.5 * (bared_m_s - bared_m_d)
            s_xy = -bared_m_11

            m_12 = m_local[5]
            m_21 = m_local[6]
            g_x = force[0,i,j,0]
            g_y = force[1,i,j,0]
            dx_sxx = dev_factor* (theta*dis_x - m_12) - g_x
            dy_syy = dev_factor*(theta*dis_y - m_21) - g_y
            dy_sxy = dev_factor*(m_12 - theta*dis_x)
            dx_sxy = dev_factor*(m_21 - theta*dis_y)

            # write macroscopics to global array
            macro = macro_vec()
            macro[0] = dis_x
            macro[1] = dis_y
            macro[2] = s_xx * L / T
            macro[3] = s_yy * L / T  # ToDo: Add kappa to rescaling!
            macro[4] = s_xy * L / T
            macro[5] = dx_sxx/T
            macro[6] = dy_syy/T
            macro[7] = dy_sxy/T
            macro[8] = dx_sxy/T

            utils.write_vec_to_global(macroscopics, macro, i, j, 9)

        '''@wp.kernel
        def unscaled_moments(f: Any, unscaled_moments: Any, L: Any, T: Any):    
            i, j, k = wp.tid()
            f_local = utils.read_local_population(f, i, j)
            m = utils.calc_moments(f_local)
            #m_10
            unscaled_moments[0, i, j] = m[0]
            #m_01
            unscaled_moments[1, i, j] = m[1]
            #m_11
            unscaled_moments[2, i, j] = m[2] * T / L
            #m_s
            unscaled_moments[3, i, j] = m[3] * T / L
            #m_d
            unscaled_moments[4, i, j] = m[4] * T / L
            #m_12
            unscaled_moments[5, i, j] = m[5] 
            #m_21
            unscaled_moments[6, i, j] = m[6] 
            #m_f
            unscaled_moments[7, i, j] = m[7] * T / L'''
        
        self.update_bared_moments_kernel = update
        self.calc_macroscopics_kernel = calc_macroscopics



    def update_bared_moments(self, f):  # updates all bared moments
        params = SimulationParams()
        theta = params.theta
        wp.launch(
            self.update_bared_moments_kernel,
            inputs=[f, self.bared_moments, self.force, self.omega, theta],
            dim=f.shape[1:],
        )

    def get_macroscopics_host(self, f):
       return self.get_macroscopics_device(f).numpy()

    def get_macroscopics_device(self, f):
        params = SimulationParams()
        T = params.T
        L = params.L
        theta = params.theta
        self.update_bared_moments(f)
        wp.launch(
            self.calc_macroscopics_kernel,
            inputs=[self.macroscopics, f, self.bared_moments, self.force, L, T, theta],
            dim=self.macroscopics.shape[1:],
        )
        return self.macroscopics

    def get_bared_moments_device(self, f):
        self.update_bared_moments(f)
        return self.bared_moments
