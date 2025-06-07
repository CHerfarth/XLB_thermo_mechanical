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


class SolidBaredMoments(Operator):
    def __init__(self, grid, omega, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
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
            f_vec: vec, force_x: self.compute_dtype, force_y: self.compute_dtype, omega: vec, theta: self.compute_dtype
        ):

            bared_m = calc_moments(f_vec)
            zero_p_five = self.compute_dtype(0.5)

            # apply half-forcing and get displacement
            bared_m[0] += -zero_p_five * force_x
            bared_m[1] += -zero_p_five * force_y 

            m_eq = calc_equilibrium(bared_m, theta)  # do something with this?
            for l in range(2, self.velocity_set.q):
                tau = (self.compute_dtype(1)/omega[l]) - self.compute_dtype(0.5)
                if not (wp.abs(omega[l] - self.compute_dtype(1)) < self.compute_dtype(1e-7)):
                    bared_m[l] = (self.compute_dtype(1) - tau*omega[l]/(self.compute_dtype(1)-omega[l]))*m_eq[l] + (tau/(self.compute_dtype(1)-omega[l]) - tau)*bared_m[l]



            return bared_m 

        @wp.kernel
        def kernel(
            bared_moments: wp.array4d(dtype=self.store_dtype),
            f: wp.array4d(dtype=self.store_dtype),
            force: wp.array4d(dtype=self.store_dtype),
            omega: vec,
            theta: self.compute_dtype,
        ):
            i, j, k = wp.tid()
            f_vec = read_local_population(f, i, j)
            force_x = self.compute_dtype(force[0,i,j,0])
            force_y = self.compute_dtype(force[1,i,j,0])

            bared_m = functional(f_vec=f_vec, force_x=force_x, force_y=force_y, omega=omega, theta=theta)

            write_population_to_global(bared_moments, bared_m, i, j)
        
        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self,  f, output_array, force):
        params = SimulationParams()
        theta = params.theta
        kappa = float(params.kappa)
        wp.launch(
            self.warp_kernel,
            inputs=[output_array, f, force, self.omega, theta],
            dim=output_array.shape[1:],
        )
        return output_array
