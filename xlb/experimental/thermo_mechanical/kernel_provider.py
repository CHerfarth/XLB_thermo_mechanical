import warp as wp
from xlb.precision_policy import PrecisionPolicy
from typing import Any
import sympy
import numpy as np
from xlb.utils import save_fields_vtk, save_image
from xlb.experimental.thermo_mechanical.solid_simulation_params import SimulationParams
from xlb import DefaultConfig


# Mapping:
#    i  j   |   f_q
#    0  0   |   0
#    0  1   |   1
#   0  -1   |   2
#    1  0   |   3
#   -1  1   |   4
#    1  -1  |   5
#   -1  0   |   6
#    1  1   |   7
#   -1  -1  |   8

# Mapping for moments:
#    i  j   |   m_q
#    1  0   |   0
#    0  1   |   1
#    1  1   |   2
#    s      |   3
#    d      |   4
#    1  2   |   5
#    2  1   |   6
#    f      |   7
#    0  0   |   8 (irrelevant)

class KernelProvider:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, precision_policy=None):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            #compile all kernels

            params = SimulationParams()
            K_scaled = params.K
            theta = params.theta
            lamb = params.lamb

            if precision_policy == None:
                precision_policy = DefaultConfig.default_precision_policy
            
            compute_dtype = precision_policy.compute_precision.wp_dtype
            store_dtype = precision_policy.store_precision.wp_dtype

            solid_vec = wp.vec(
                9, dtype=compute_dtype
            )  

            self.solid_vec = solid_vec

            @wp.func
            def read_local_population(f: wp.array4d(dtype=store_dtype), x: wp.int32, y: wp.int32):
                f_local = solid_vec()
                for i in range(9):
                    f_local[i] = compute_dtype(f[i, x, y, 0])
                return f_local
            


            @wp.func
            def write_population_to_global(f: wp.array4d(dtype=store_dtype), f_local: solid_vec, x: wp.int32, y: wp.int32):
                for i in range(9):
                    f[i, x, y, 0] = store_dtype(f_local[i])


            @wp.func
            def write_vec_to_global(array: wp.array4d(dtype=store_dtype), array_local: Any, x: wp.int32, y: wp.int32, dim: wp.int32):
                for i in range(dim):
                    array[i, x, y, 0] = store_dtype(array_local[i])


            @wp.func
            def calc_moments(f: solid_vec):
                m = solid_vec()
                # Todo: find better way to do this!
                m[0] = f[3] - f[6] + f[7] - f[4] - f[8] + f[5]
                m[1] = f[1] - f[2] + f[7] + f[4] - f[8] - f[5]
                m[2] = f[7] - f[4] + f[8] - f[5]
                m[3] = f[3] + f[1] + f[6] + f[2] + 2.0 * f[7] + 2.0 * f[4] + 2.0 * f[8] + 2.0 * f[5]
                m[4] = f[3] - f[1] + f[6] - f[2]
                m[5] = f[7] - f[4] - f[8] + f[5]
                m[6] = f[7] + f[4] - f[8] - f[5]
                m[7] = f[7] + f[4] + f[8] + f[5]
                m[8] = 0.0
                # m_7 is m_22 right now, we now convert it to m_f
                tau_s = 2.0 * K_scaled / (1.0 + theta)
                tau_f = 0.5  # todo: make modular, as function argument etc
                gamma = (theta * tau_f) / ((1.0 + theta) * (tau_s - tau_f))
                m[7] += gamma * m[3]
                return m


            @wp.kernel
            def copy_populations(origin: wp.array4d(dtype=store_dtype), dest: wp.array4d(dtype=store_dtype), dim: wp.int32):
                i, j, k = wp.tid()
                for l in range(dim):
                    dest[l, i, j, 0] = origin[l, i, j, 0]

            @wp.kernel
            def set_population_to_zero(f: wp.array4d(dtype=store_dtype), dim: Any):
                i,j,k = wp.tid()
                for l in range(dim):
                    f[l, i, j, 0] = 0.

            @wp.kernel
            def multiply_populations(f: wp.array4d(dtype=store_dtype), factor: compute_dtype, dim: wp.int32):
                i, j, k = wp.tid()
                for l in range(dim):
                    f[l, i, j, 0] = store_dtype(factor*compute_dtype(f[l,i,j,0]))


            @wp.kernel
            def subtract_populations(a: wp.array4d(dtype=store_dtype), b: wp.array4d(dtype=store_dtype), c: wp.array4d(dtype=store_dtype), dim: wp.int32):
                i, j, k = wp.tid()
                for l in range(dim):
                    c[l, i, j, 0] = store_dtype(compute_dtype(a[l, i, j, 0]) - compute_dtype(b[l, i, j, 0]))


            @wp.kernel
            def add_populations(a: wp.array4d(dtype=store_dtype), b: wp.array4d(dtype=store_dtype), c: wp.array4d(dtype=store_dtype), dim: wp.int32):
                i, j, k = wp.tid()
                for l in range(dim):
                    c[l, i, j, 0] = store_dtype(compute_dtype(a[l, i, j, 0]) + compute_dtype(b[l, i, j, 0]))


            @wp.func
            def calc_populations(m: solid_vec):
                f = solid_vec()
                # m_7 is m_f right now, we convert it back to m_22
                tau_s = 2.0 * K_scaled / (1.0 + theta)
                tau_f = 0.5  # todo: make modular, as function argument etc
                gamma = (theta * tau_f) / ((1.0 + theta) * (tau_s - tau_f))
                m[7] += -gamma * m[3]
                # Todo: find better way to do this!
                f[3] = 2.0 * m[0] + m[3] + m[4] - 2.0 * m[5] - 2.0 * m[7]
                f[1] = 2.0 * m[1] + m[3] - m[4] - 2.0 * m[6] - 2.0 * m[7]
                f[6] = -2.0 * m[0] + m[3] + m[4] + 2.0 * m[5] - 2.0 * m[7]
                f[2] = -2.0 * m[1] + m[3] - m[4] + 2.0 * m[6] - 2.0 * m[7]
                f[7] = m[2] + m[5] + m[6] + m[7]
                f[4] = -m[2] - m[5] + m[6] + m[7]
                f[8] = m[2] - m[5] - m[6] + m[7]
                f[5] = -m[2] + m[5] - m[6] + m[7]
                f[0] = 0.0
                for i in range(9):
                    f[i] = f[i] / 4.0
                return f


            @wp.func
            def calc_equilibrium(m: solid_vec, theta: Any):
                m_eq = solid_vec()
                m_eq[0] = m[0]
                m_eq[1] = m[1]
                m_eq[2] = 0.0
                m_eq[3] = 0.0
                m_eq[4] = 0.0
                m_eq[5] = theta * m[0]
                m_eq[6] = theta * m[1]
                m_eq[7] = 0.0
                m_eq[8] = 0.0
                return m_eq
            
            @wp.kernel
            def relaxation(f_after_stream: wp.array4d(dtype=store_dtype), f_previous: wp.array4d(dtype=store_dtype), defect_correction: wp.array4d(dtype=store_dtype), f_destination: wp.array4d(dtype=store_dtype), gamma: compute_dtype, dim: wp.int32):
                i, j, k = wp.tid()
                for l in range(dim):
                    f_destination[l, i, j, 0] = store_dtype(
                        gamma * (compute_dtype(f_after_stream[l, i, j, 0]) - compute_dtype(defect_correction[l, i, j, 0])) + (1.0 - gamma) * compute_dtype(f_previous[l, i, j, 0])
                    )


            @wp.kernel
            def interpolate(fine: wp.array4d(dtype=store_dtype), coarse: wp.array4d(dtype=store_dtype), dim: wp.int32):
                i,j,k = wp.tid()
                coarse_i = i/2
                coarse_j = j/2 #check if really rounds down!

                if (wp.mod(i, 2) == 0) and (wp.mod(j, 2) == 0) or True:

                    for l in range(dim):
                        fine[l,i,j,0] = coarse[l,coarse_i,coarse_j,0]

            

            @wp.kernel
            def restrict(coarse: wp.array4d(dtype=store_dtype), fine: wp.array4d(dtype=store_dtype), fine_nodes_x: wp.int32, fine_nodes_y: wp.int32, dim: wp.int32):
                i,j,k = wp.tid()

                for l in range(dim):
                        val =  0.
                        val += fine[l, 2*i, 2*j, 0]
                        val += fine[l, 2*i+1, 2*j, 0]
                        val += fine[l, 2*i, 2*j+1, 0]
                        val += fine[l, 2*i+1, 2*j+1, 0]
                        coarse[l, i, j, 0] = 0.25*val


            # Set all declared functions as properties of the class
            self.read_local_population = read_local_population
            self.write_population_to_global = write_population_to_global
            self.write_vec_to_global = write_vec_to_global
            self.calc_moments = calc_moments
            self.copy_populations = copy_populations
            self.set_population_to_zero = set_population_to_zero
            self.multiply_populations = multiply_populations
            self.subtract_populations = subtract_populations
            self.add_populations = add_populations
            self.calc_populations = calc_populations
            self.calc_equilibrium = calc_equilibrium
            self.relaxation = relaxation
            self.interpolate = interpolate
            self.restrict = restrict
