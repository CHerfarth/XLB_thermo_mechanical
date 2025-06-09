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
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider


class SolidMacroscopics(Operator):
    def __init__(self, grid, omega, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        self.omega = omega
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

    def _construct_warp(self):
        # get warp funcs
        kernel_provider = KernelProvider()
        vec = kernel_provider.vec
        read_local_population = kernel_provider.read_local_population
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global
        write_vec_to_global = kernel_provider.write_vec_to_global

        @wp.func
        def functional(
            bared_m: vec,
            force_x: self.compute_dtype,
            force_y: self.compute_dtype,
            omega: vec,
            theta: self.compute_dtype,
            L: self.compute_dtype,
            T: self.compute_dtype,
            kappa: self.compute_dtype,
        ):
            # --------Step 1: calculate macroscopics------------------
            tau_t = self.compute_dtype(0.5)
            dev_factor = self.compute_dtype(2.0) / (
                self.compute_dtype(1.0) + self.compute_dtype(2.0) * tau_t
            )
            # displacement
            dis_x = bared_m[0]
            dis_y = bared_m[1]
            # stress
            bared_m_s = bared_m[3]
            bared_m_d = bared_m[4]
            bared_m_11 = bared_m[2]
            s_xx = -self.compute_dtype(0.5) * (bared_m_s + bared_m_d)
            s_yy = -self.compute_dtype(0.5) * (bared_m_s - bared_m_d)
            s_xy = -bared_m_11
            # stress derivatives
            m_12 = bared_m[5]
            m_21 = bared_m[6]
            dx_sxx = dev_factor * (theta * dis_x - m_12) - force_x
            dy_syy = dev_factor * (theta * dis_y - m_21) - force_y
            dy_sxy = dev_factor * (m_12 - theta * dis_x)
            dx_sxy = dev_factor * (m_21 - theta * dis_y)
            ##--------Step 3: rescale back to physical units
            macro = vec()
            macro[0] = dis_x
            macro[1] = dis_y
            macro[2] = s_xx * (L * kappa) / T
            macro[3] = s_yy * (L * kappa) / T
            macro[4] = s_xy * (L * kappa) / T
            macro[5] = dx_sxx * kappa / T
            macro[6] = dy_syy * kappa / T
            macro[7] = dy_sxy * kappa / T
            macro[8] = dx_sxy * kappa / T

            return macro

        @wp.kernel
        def kernel(
            macroscopics: wp.array4d(dtype=self.store_dtype),
            bared_moments: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            L: self.compute_dtype,
            T: self.compute_dtype,
            theta: self.compute_dtype,
            kappa: self.compute_dtype,
        ):
            i, j, k = wp.tid()
            bared_m = read_local_population(bared_moments, i, j)
            force_x = self.compute_dtype(force[0, i, j, 0])
            force_y = self.compute_dtype(force[1, i, j, 0])

            macro = functional(
                bared_m=bared_m,
                force_x=force_x,
                force_y=force_y,
                omega=omega,
                theta=theta,
                L=L,
                T=T,
                kappa=kappa,
            )

            write_population_to_global(macroscopics, macro, i, j)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bared_moments, output_array, force):
        params = SimulationParams()
        T = params.T
        L = params.L
        theta = params.theta
        kappa = float(params.kappa)
        wp.launch(
            self.warp_kernel,
            inputs=[output_array, bared_moments, force, self.omega, L, T, theta, kappa],
            dim=output_array.shape[1:],
        )
        return output_array
