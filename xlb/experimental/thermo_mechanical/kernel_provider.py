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
    def __init__(self, precision_policy=None):
        self._initialized = True
        # compile all kernels

        if precision_policy == None:
            precision_policy = DefaultConfig.default_precision_policy

        velocity_set = DefaultConfig.velocity_set

        compute_dtype = precision_policy.compute_precision.wp_dtype
        store_dtype = precision_policy.store_precision.wp_dtype

        params = SimulationParams()
        K_scaled = compute_dtype(params.K)
        theta = compute_dtype(params.theta)
        lamb = compute_dtype(params.lamb)

        solid_vec = wp.vec(9, dtype=compute_dtype)

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
            m[3] = (
                f[3]
                + f[1]
                + f[6]
                + f[2]
                + compute_dtype(2.0) * f[7]
                + compute_dtype(2.0) * f[4]
                + compute_dtype(2.0) * f[8]
                + compute_dtype(2.0) * f[5]
            )
            m[4] = f[3] - f[1] + f[6] - f[2]
            m[5] = f[7] - f[4] - f[8] + f[5]
            m[6] = f[7] + f[4] - f[8] - f[5]
            m[7] = f[7] + f[4] + f[8] + f[5]
            m[8] = compute_dtype(0.0)
            # m_7 is m_22 right now, we now convert it to m_f
            tau_s = compute_dtype(2.0) * K_scaled / (compute_dtype(1.0) + theta)
            tau_f = compute_dtype(0.5)  # todo: make modular, as function argument etc
            gamma = (theta * tau_f) / ((compute_dtype(1.0) + theta) * (tau_s - tau_f))
            m[7] += gamma * m[3]
            return m

        @wp.kernel
        def copy_populations(origin: wp.array4d(dtype=store_dtype), dest: wp.array4d(dtype=store_dtype), dim: wp.int32):
            i, j, k = wp.tid()
            for l in range(dim):
                dest[l, i, j, 0] = origin[l, i, j, 0]

        @wp.kernel
        def set_population_to_zero(f: wp.array4d(dtype=store_dtype), dim: Any):
            i, j, k = wp.tid()
            for l in range(dim):
                f[l, i, j, 0] = store_dtype(0.0)

        @wp.kernel
        def multiply_populations(f: wp.array4d(dtype=store_dtype), factor: compute_dtype, dim: wp.int32):
            i, j, k = wp.tid()
            for l in range(dim):
                f[l, i, j, 0] = store_dtype(factor * compute_dtype(f[l, i, j, 0]))

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

        @wp.kernel
        def add_populations_ignore_nans(
            a: wp.array4d(dtype=store_dtype), b: wp.array4d(dtype=store_dtype), c: wp.array4d(dtype=store_dtype), dim: wp.int32
        ):
            i, j, k = wp.tid()
            for l in range(dim):
                a_val = compute_dtype(a[l, i, j, 0])
                b_val = compute_dtype(b[l, i, j, 0])
                if a_val == compute_dtype(wp.nan):
                    a_val = compute_dtype(0.0)
                if b_val == compute_dtype(wp.nan):
                    b_val = compute_dtype(0.0)
                c[l, i, j, 0] = store_dtype(a_val + b_val)


        @wp.func
        def calc_populations(m: solid_vec):
            f = solid_vec()
            # m_7 is m_f right now, we convert it back to m_22
            tau_s = compute_dtype(2.0) * K_scaled / (compute_dtype(1.0) + theta)
            tau_f = compute_dtype(0.5)  # todo: make modular, as function argument etc
            gamma = (theta * tau_f) / ((compute_dtype(1.0) + theta) * (tau_s - tau_f))
            m[7] += -gamma * m[3]
            # Todo: find better way to do this!
            two = compute_dtype(2.0)
            f[3] = two * m[0] + m[3] + m[4] - two * m[5] - two * m[7]
            f[1] = two * m[1] + m[3] - m[4] - two * m[6] - two * m[7]
            f[6] = -two * m[0] + m[3] + m[4] + two * m[5] - two * m[7]
            f[2] = -two * m[1] + m[3] - m[4] + two * m[6] - two * m[7]
            f[7] = m[2] + m[5] + m[6] + m[7]
            f[4] = -m[2] - m[5] + m[6] + m[7]
            f[8] = m[2] - m[5] - m[6] + m[7]
            f[5] = -m[2] + m[5] - m[6] + m[7]
            f[0] = compute_dtype(0.0)
            for i in range(9):
                f[i] = f[i] / compute_dtype(4.0)
            return f

        @wp.func
        def calc_equilibrium(m: solid_vec, theta: compute_dtype):
            zero = compute_dtype(0.0)
            m_eq = solid_vec()
            m_eq[0] = m[0]
            m_eq[1] = m[1]
            m_eq[2] = zero
            m_eq[3] = zero
            m_eq[4] = zero
            m_eq[5] = theta * m[0]
            m_eq[6] = theta * m[1]
            m_eq[7] = zero
            m_eq[8] = zero
            return m_eq

        @wp.kernel
        def relaxation(
            f_after_stream: wp.array4d(dtype=store_dtype),
            f_previous: wp.array4d(dtype=store_dtype),
            defect_correction: wp.array4d(dtype=store_dtype),
            f_destination: wp.array4d(dtype=store_dtype),
            gamma: compute_dtype,
            dim: wp.int32,
        ):
            i, j, k = wp.tid()
            for l in range(dim):
                f_destination[l, i, j, 0] = store_dtype(
                    gamma * (compute_dtype(f_after_stream[l, i, j, 0]) - compute_dtype(defect_correction[l, i, j, 0]))
                    + (compute_dtype(1.0) - gamma) * compute_dtype(f_previous[l, i, j, 0])
                )

        @wp.kernel
        def relaxation_no_defect(
            f_after_stream: wp.array4d(dtype=store_dtype),
            f_previous: wp.array4d(dtype=store_dtype),
            f_destination: wp.array4d(dtype=store_dtype),
            gamma: compute_dtype,
            dim: wp.int32,
        ):
            i, j, k = wp.tid()
            for l in range(dim):
                f_destination[l, i, j, 0] = store_dtype(
                    gamma * (compute_dtype(f_after_stream[l, i, j, 0]))
                    + (compute_dtype(1.0) - gamma) * compute_dtype(f_previous[l, i, j, 0])
                )



        @wp.func
        def local_contains_nan(local: solid_vec):
            for l in range(velocity_set.q):
                if wp.isnan(local[l]):
                    return True
            return False

        @wp.kernel
        def interpolate_through_macroscopics(
            fine: wp.array4d(dtype=store_dtype), macroscopics_coarse: wp.array4d(dtype=store_dtype), coarse_nodes_x: wp.int32, coarse_nodes_y: wp.int32, L: compute_dtype, T: compute_dtype
        ):
            i, j, k = wp.tid()

            coarse_i = i / 2
            coarse_j = j / 2  # rounds down

            res_i = i - coarse_i * 2
            res_j = j - coarse_j * 2

            macr_a = read_local_population(macroscopics_coarse, coarse_i, coarse_j)
            macr_b = macr_a
            macr_c = macr_a
            macr_d = macr_a

            # Coding: f_a closest coarsepoint to new fine point
            #  f_b, f_c along edges of coarse square
            #  f_d along diagonal

            shift_x = 0
            shift_y = 0

            if res_i == 0 and res_j == 0:
                shift_x = -1
                shift_y = -1
            elif res_i == 0 and res_j == 1:
                shift_x = -1
                shift_y = 1
            elif res_i == 1 and res_j == 0:
                shift_x = 1
                shift_y = -1
            else:
                shift_x = 1
                shift_y = 1
            

            macr_b = read_local_population(macroscopics_coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), coarse_j)
            macr_c = read_local_population(macroscopics_coarse, coarse_i, wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y))
            macr_d = read_local_population(
                macroscopics_coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y)
            )


            macr  = compute_dtype(0.0625) * (
                compute_dtype(9.0) * macr_a + compute_dtype(3.0) * macr_b + compute_dtype(3.0) * macr_c + compute_dtype(1.0) * macr_d
            )

            u_x = macr[0]
            u_y = macr[1]
            s_xx = macr[2] * T / L
            s_yy = macr[3] * T / L
            s_xy = macr[4] * T / L
            s_s = s_xx + s_yy
            s_d = s_xx - s_yy
            tau_11 = compute_dtype(0.5)
            tau_s = compute_dtype(0.5)
            tau_d = compute_dtype(0.5)
            tau_f = compute_dtype(0.5)
            theta = compute_dtype(1./3.)
            one = compute_dtype(1)
            two = compute_dtype(2)
            
            m_fine = solid_vec()
            m_fine[0] = u_x *compute_dtype(0)
            m_fine[1] = u_y *compute_dtype(0)
            m_fine[2] = -(one + one/(two*tau_11))*s_xy*compute_dtype(0)
            m_fine[3] = -(one + one/(two*tau_s))*s_s*compute_dtype(0)
            m_fine[4] = -(one + one/(two*tau_d))*s_d*compute_dtype(0)
            m_fine[5] = theta*u_x*compute_dtype(0)
            m_fine[6] = theta*u_y*compute_dtype(0)
            m_fine[7] = (-theta*(one+two*tau_f))/((one+theta)*(tau_s-tau_f))*s_s*compute_dtype(0)
            m_fine[8] = compute_dtype(0.)*compute_dtype(0)

            f_local_fine = calc_populations(m_fine)
            write_population_to_global(fine, f_local_fine, i, j)
        
        @wp.kernel
        def interpolate_through_moments_no_boundaries(
            fine: wp.array4d(dtype=store_dtype), coarse: wp.array4d(dtype=store_dtype), coarse_nodes_x: wp.int32, coarse_nodes_y: wp.int32
        ):
            i, j, k = wp.tid()

            coarse_i = i / 2
            coarse_j = j / 2  # rounds down

            res_i = i - coarse_i * 2
            res_j = j - coarse_j * 2

            f_a = read_local_population(coarse, coarse_i, coarse_j)
            f_b = f_a
            f_c = f_a
            f_d = f_a

            # Coding: f_a closest coarsepoint to new fine point
            #  f_b, f_c along edges of coarse square
            #  f_d along diagonal

            shift_x = 0
            shift_y = 0

            if res_i == 0 and res_j == 0:
                shift_x = -1
                shift_y = -1
            elif res_i == 0 and res_j == 1:
                shift_x = -1
                shift_y = 1
            elif res_i == 1 and res_j == 0:
                shift_x = 1
                shift_y = -1
            else:
                shift_x = 1
                shift_y = 1

            f_b = read_local_population(coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), coarse_j)
            f_c = read_local_population(coarse, coarse_i, wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y))
            f_d = read_local_population(
                coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y)
            )

            m_a = calc_moments(f_a)
            m_b = calc_moments(f_b)
            m_c = calc_moments(f_c)
            m_d = calc_moments(f_d)

            m_fine = compute_dtype(0.0625)*(compute_dtype(9.)*m_a + compute_dtype(3.)*m_b + compute_dtype(3.)*m_c + compute_dtype(1.)*m_d)

            # scale necessary components of m
            '''m_fine[0] = compute_dtype(1)*m_fine[0] 
            m_fine[1] = compute_dtype(1)*m_fine[1] 
            m_fine[2] = compute_dtype(0.5)*m_fine[2] 
            m_fine[3] = compute_dtype(0.5)*m_fine[3] 
            m_fine[4] = compute_dtype(0.5)*m_fine[4] 
            m_fine[5] = compute_dtype(1)*m_fine[5] 
            m_fine[6] = compute_dtype(1)*m_fine[6] 
            m_fine[7] = compute_dtype(0.5)*m_fine[7] 
            m_fine[8] = compute_dtype(1)*m_fine[8]''' 

            f_local_fine = calc_populations(m_fine)
            write_population_to_global(fine, f_local_fine, i, j)

        @wp.kernel
        def interpolate_through_moments_with_boundaries(
            fine: wp.array4d(dtype=store_dtype), coarse: wp.array4d(dtype=store_dtype), coarse_nodes_x: wp.int32, coarse_nodes_y: wp.int32, coarse_boundary_array: wp.array4d(dtype=wp.int8)
        ):
            i, j, k = wp.tid()

            coarse_i = i / 2
            coarse_j = j / 2  # rounds down

            res_i = i - coarse_i * 2
            res_j = j - coarse_j * 2

            f_a = read_local_population(coarse, coarse_i, coarse_j)
            f_b = f_a
            f_c = f_a
            f_d = f_a

            # Coding: f_a closest coarsepoint to new fine point
            #  f_b, f_c along edges of coarse square
            #  f_d along diagonal

            shift_x = 0
            shift_y = 0

            if res_i == 0 and res_j == 0:
                shift_x = -1
                shift_y = -1
            elif res_i == 0 and res_j == 1:
                shift_x = -1
                shift_y = 1
            elif res_i == 1 and res_j == 0:
                shift_x = 1
                shift_y = -1
            else:
                shift_x = 1
                shift_y = 1

            f_b = read_local_population(coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), coarse_j)
            f_c = read_local_population(coarse, coarse_i, wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y))
            f_d = read_local_population(
                coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y)
            )

            m_a = calc_moments(f_a)
            m_b = calc_moments(f_b)
            m_c = calc_moments(f_c)
            m_d = calc_moments(f_d)

            domain_a, domain_b, domain_c, domain_d = True, True, True, True

            # check for boundary 
            if (coarse_boundary_array[0, coarse_i, coarse_j, 0] == wp.int8(0)):
                domain_a = False  
            if (coarse_boundary_array[0, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), coarse_j, 0] == wp.int8(0)):
                domain_b = False
            if (coarse_boundary_array[0, coarse_i, wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y), 0] == wp.int8(0)):
                domain_c = False  
            if (coarse_boundary_array[0, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y), 0] == wp.int8(0)):
                domain_d = False  
          

            if (domain_a and domain_b and domain_c and domain_d) or False:
                m_fine = compute_dtype(0.0625)*(compute_dtype(9.)*m_a + compute_dtype(3.)*m_b + compute_dtype(3.)*m_c + compute_dtype(1.)*m_d)
            elif domain_a:
                m_fine = m_a
            elif domain_b:
                m_fine = m_b
            elif domain_c:
                m_fine = m_c
            elif domain_d:
                m_fine = m_d
            else:
                m_fine = solid_vec()
                for l in range(velocity_set.q):
                    m_fine[l] = compute_dtype(wp.nan)


            # scale necessary components of m
            m_fine[0] = compute_dtype(1)*m_fine[0] 
            m_fine[1] = compute_dtype(1)*m_fine[1] 
            m_fine[2] = compute_dtype(0.5)*m_fine[2] 
            m_fine[3] = compute_dtype(0.5)*m_fine[3] 
            m_fine[4] = compute_dtype(0.5)*m_fine[4] 
            m_fine[5] = compute_dtype(1)*m_fine[5] 
            m_fine[6] = compute_dtype(1)*m_fine[6] 
            m_fine[7] = compute_dtype(0.5)*m_fine[7] 
            m_fine[8] = compute_dtype(1)*m_fine[8] 

            f_local_fine = calc_populations(m_fine)
            write_population_to_global(fine, f_local_fine, i, j)

        @wp.kernel
        def interpolate(fine: wp.array4d(dtype=store_dtype), coarse: wp.array4d(dtype=store_dtype), dim: wp.int32):
            i, j, k = wp.tid()
            coarse_i = i / 2
            coarse_j = j / 2  # check if really rounds down!

            for l in range(dim):
                fine[l, i, j, 0] = coarse[l, coarse_i, coarse_j, 0]

        @wp.kernel
        def check_for_nans(f: wp.array4d(dtype=store_dtype), boundary_array: wp.array4d(dtype=wp.int8)):
            i, j, k = wp.tid()
            f_local = read_local_population(f, i, j)
            if boundary_array[0, i, j, 0] != wp.int8(0):
                for l in range(velocity_set.q):
                    assert not wp.isnan(f_local[l])
                
        @wp.kernel
        def restrict_no_boundaries(
            coarse: wp.array4d(dtype=store_dtype), fine: wp.array4d(dtype=store_dtype)
        ):
            i, j, k = wp.tid()

            f_a = read_local_population(fine, 2 * i, 2 * j)
            f_b = read_local_population(fine, 2 * i + 1, 2 * j)
            f_c = read_local_population(fine, 2 * i, 2 * j + 1)
            f_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)

            coarse_f = compute_dtype(0.25)*(f_a + f_b + f_c + f_d)

            write_population_to_global(coarse, coarse_f, i, j)

        @wp.kernel
        def restrict_with_boundaries(
            coarse: wp.array4d(dtype=store_dtype), fine: wp.array4d(dtype=store_dtype), fine_boundary_array: wp.array4d(dtype=wp.int8)
        ):
            i, j, k = wp.tid()

            f_a = read_local_population(fine, 2 * i, 2 * j)
            f_b = read_local_population(fine, 2 * i + 1, 2 * j)
            f_c = read_local_population(fine, 2 * i, 2 * j + 1)
            f_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)

            domain_a = True
            domain_b = True
            domain_c = True
            domain_d = True

            if fine_boundary_array[0, 2 * i, 2 * j, 0] == wp.int8(0):
                domain_a = False
            if fine_boundary_array[0, 2 * i + 1, 2 * j, 0] == wp.int8(0):
                domain_b = False
            if fine_boundary_array[0, 2 * i, 2 * j + 1, 0] == wp.int8(0):
                domain_c = False
            if fine_boundary_array[0, 2 * i + 1, 2 * j + 1, 0] == wp.int8(0):
                domain_d = False
            
            coarse_f = solid_vec()
            for l in range(velocity_set.q):
                coarse_f[l] = compute_dtype(0)
            
            count = compute_dtype(0)
            if domain_a:
                coarse_f += f_a
                count += compute_dtype(1)
            if domain_b:
                coarse_f += f_b
                count += compute_dtype(1)
            if domain_c:
                coarse_f += f_c
                count += compute_dtype(1)
            if domain_d:
                coarse_f += f_d
                count += compute_dtype(1)
            
            if not (count < compute_dtype(1e-6)):
                coarse_f /= count
            else:
                for l in range(velocity_set.q):
                    coarse_f[l] = compute_dtype(wp.nan)

            write_population_to_global(coarse, coarse_f, i, j)


        @wp.kernel
        def restrict_through_moments(coarse: wp.array4d(dtype=store_dtype), fine: wp.array4d(dtype=store_dtype)):
            i, j, k = wp.tid()

            f_local_fine_a = read_local_population(fine, 2 * i, 2 * j)
            f_local_fine_b = read_local_population(fine, 2 * i + 1, 2 * j)
            f_local_fine_c = read_local_population(fine, 2 * i, 2 * j + 1)
            f_local_fine_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)
            m_fine_a = calc_moments(f_local_fine_a)
            m_fine_b = calc_moments(f_local_fine_b)
            m_fine_c = calc_moments(f_local_fine_c)
            m_fine_d = calc_moments(f_local_fine_d)

            m_coarse = compute_dtype(0.25) * (m_fine_a + m_fine_b + m_fine_c + m_fine_d)

            # scale necessary components of m
            #m_coarse[2] = compute_dtype(0.125) * m_coarse[2]
            #m_coarse[3] = compute_dtype(0.125) * m_coarse[3]
            #m_coarse[4] = compute_dtype(0.125) * m_coarse[4]
            #m_coarse[7] = compute_dtype(0.125) * m_coarse[7]
            
            f_local_coarse = calc_populations(m_coarse)
            write_population_to_global(coarse, f_local_coarse, i, j)

        @wp.kernel
        def l2_norm(f: wp.array4d(dtype=store_dtype), sq_norm: wp.array(dtype=compute_dtype)):
            i, j, k = wp.tid()

            f_local = read_local_population(f, i, j)

            local_norm = compute_dtype(0.0)
            for l in range(velocity_set.q):
                local_norm += compute_dtype(f_local[l]) * compute_dtype(f_local[l])

            wp.atomic_add(sq_norm, 0, local_norm)

        @wp.kernel
        def convert_moments_to_populations(m: wp.array4d(dtype=store_dtype), f: wp.array4d(dtype=store_dtype)):
            i, j, k = wp.tid()

            m_local = read_local_population(m, i, j)
            f_local = calc_populations(m_local)
            write_population_to_global(f, f_local, i, j)

        @wp.kernel
        def convert_populations_to_moments(f: wp.array4d(dtype=store_dtype), m: wp.array4d(dtype=store_dtype)):
            i, j, k = wp.tid()

            f_local = read_local_population(f, i, j)
            m_local = calc_moments(f_local)
            write_population_to_global(m, m_local, i, j)

        @wp.kernel
        def set_zero_outside_boundary(
            f: wp.array4d(dtype=store_dtype),
            boundary_array: wp.array4d(dtype=wp.int8),
        ):
            i, j, k = wp.tid()  # for 2d k will equal 1
            if boundary_array[0, i, j, 0] == wp.int8(0):  # if outside domain, just set to 0
                for l in range(velocity_set.q):
                    f[l, i, j, 0] = store_dtype(wp.nan)

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
        self.relaxation_no_defect = relaxation_no_defect
        self.interpolate = interpolate
        self.interpolate_through_moments_no_boundaries = interpolate_through_moments_no_boundaries
        self.interpolate_through_moments_with_boundaries = interpolate_through_moments_with_boundaries
        self.interpolate_through_macroscopics = interpolate_through_macroscopics
        self.restrict_no_boundaries = restrict_no_boundaries
        self.restrict_with_boundaries = restrict_with_boundaries
        self.restrict_through_moments = restrict_through_moments
        self.l2_norm = l2_norm
        self.convert_moments_to_populations = convert_moments_to_populations
        self.convert_populations_to_moments = convert_populations_to_moments
        self.set_zero_outside_boundary = set_zero_outside_boundary
        self.check_for_nans = check_for_nans