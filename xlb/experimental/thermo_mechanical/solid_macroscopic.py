from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.operator import Operator
import xlb.experimental.thermo_mechanical.solid_utils as utils


class SolidMacroscopics(Operator):
    def __init__(self, grid, force, omega, theta, L, T, boundaries=None, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        self.force = force
        self.omega = omega
        self.theta = theta
        self.boundaries = boundaries
        self.L = L
        self.T = T
        self.bared_moments = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        #Mapping for macroscopics:
        # 0: dis_x
        # 1: dis_y
        # 2: s_xx
        # 3: s_yy
        # 4: s_xy
        self.macroscopics = grid.create_field(cardinality=5, dtype=self.precision_policy.store_precision)
        if self.boundaries == None:
            self.boundaries = grid.create_field(cardinality=10, dtype=precision_policy.store_precision, fill_value=1)

    def _construct_warp(self):

        macro_vec = wp.vec(5, dtype=self.compute_dtype)

        @wp.kernel
        def update(f: Any, bared_moments: Any, force: Any, omega: Any, boundaries: Any, theta: Any):
            i, j, k = wp.tid()  # for 2d, k will equal 1

            # calculate moments
            f_local = utils.read_local_population(f, i, j)
            m = utils.calc_moments(f_local)

            # apply half-forcing and get displacement
            m[0] += 0.5 * force[0, i, j, 0]
            m[1] += 0.5 * force[1, i, j, 0]

            m_eq = utils.calc_equilibrium(m, theta)  # do something with this?
            for l in range(m._length_): 
                m[l] = 0.5*omega[l]*m_eq[l] + (1.-0.5*omega[l])*m[l]
            
            utils.write_population_to_global(bared_moments, m, i, j)
        
        @wp.kernel
        def write_out(macroscopics: Any, bared_moments: Any, L:Any, T: Any):
            i, j, k = wp.tid()
            bared_m_local = utils.read_local_population(bared_moments, i, j)

            #calculate macroscopics
            dis_x = bared_m_local[0]
            dis_y = bared_m_local[1]
            bared_m_s = bared_m_local[3]
            bared_m_d = bared_m_local[4]
            bared_m_11 = bared_m_local[2]
            s_xx = -0.5*(bared_m_s + bared_m_d)
            s_yy = -0.5*(bared_m_s - bared_m_d)
            s_xy = -bared_m_11

            #write macroscopics to global array
            macro = macro_vec()
            macro[0] = dis_x
            macro[1] = dis_y
            macro[2] = s_xx * L / T
            macro[3] = s_yy * L / T #ToDo: Add kappa to rescaling!
            macro[4] = s_xy * L / T
            utils.write_vec_to_global(macroscopics, macro, i, j, 5)



        return None, (update, write_out)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f): #updates all bared moments
        wp.launch(
            self.warp_kernel[0],
            inputs=[f, self.bared_moments, self.force, self.omega, self.boundaries, self.theta],
            dim=f.shape[1:],
        )

    def get_macroscopics_host(self):
       wp.launch(
            self.warp_kernel[1],
            inputs=[self.macroscopics, self.bared_moments, self.L, self.T],
            dim=self.macroscopics.shape[1:],
       ) 
       return self.macroscopics.numpy()

    def get_macroscopics_device(self):
       wp.launch(
            self.warp_kernel[1],
            inputs=[self.macroscopics, self.bared_moments, self.L, self.T],
            dim=self.macroscopics.shape[1:],
       ) 
       return self.macroscopics

    def get_bared_moments_device(self):
        return self.bared_moments
