from xlb.operator import Operator
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
from xlb.compute_backend import ComputeBackend
import warp as wp


class Prolongation(Operator):
    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    def _construct_warp(self):
        kernel_provider = KernelProvider()
        vec = kernel_provider.vec
        calc_moments = kernel_provider.calc_moments
        calc_equilibrium = kernel_provider.calc_equilibrium
        calc_populations = kernel_provider.calc_populations
        write_population_to_global = kernel_provider.write_population_to_global
        read_local_population = kernel_provider.read_local_population

        @wp.func
        def functional(f_a: vec, f_b: vec, f_c: vec, f_d: vec):
            m_a = calc_moments(f_a)
            m_b = calc_moments(f_b)
            m_c = calc_moments(f_c)
            m_d = calc_moments(f_d)

            m_out = self.compute_dtype(0.0625) * (
                self.compute_dtype(9.0) * m_a
                + self.compute_dtype(3.0) * m_b
                + self.compute_dtype(3.0) * m_c
                + self.compute_dtype(1.0) * m_d
            )

            # scale necessary components of m
            m_out[0] = self.compute_dtype(1) * m_out[0]
            m_out[1] = self.compute_dtype(1) * m_out[1]
            m_out[2] = self.compute_dtype(0.5) * m_out[2]*self.compute_dtype(2)
            m_out[3] = self.compute_dtype(0.5) * m_out[3]*self.compute_dtype(2)
            m_out[4] = self.compute_dtype(0.5) * m_out[4]*self.compute_dtype(2)
            m_out[5] = self.compute_dtype(1) * m_out[5]
            m_out[6] = self.compute_dtype(1) * m_out[6]
            m_out[7] = self.compute_dtype(0.5) * m_out[7]*self.compute_dtype(2)
            m_out[8] = self.compute_dtype(1) * m_out[8]

            f_out = calc_populations(m_out)

            return f_out

        @wp.kernel
        def kernel(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
            coarse_nodes_x: wp.int32,
            coarse_nodes_y: wp.int32,
        ):
            i, j, k = wp.tid()

            coarse_i = i / 2
            coarse_j = j / 2  # rounds down

            res_i = i - coarse_i * 2
            res_j = j - coarse_j * 2

            _f_a = read_local_population(coarse, coarse_i, coarse_j)
            _f_b = _f_a
            _f_c = _f_a
            _f_d = _f_a

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

            _f_b = read_local_population(
                coarse, wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x), coarse_j
            )
            _f_c = read_local_population(
                coarse, coarse_i, wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y)
            )
            _f_d = read_local_population(
                coarse,
                wp.mod(coarse_i + shift_x + coarse_nodes_x, coarse_nodes_x),
                wp.mod(coarse_j + shift_y + coarse_nodes_y, coarse_nodes_y),
            )

            _error_approx = functional(f_a=_f_a, f_b=_f_b, f_c=_f_c, f_d=_f_d)
            _f_old = read_local_population(fine, i, j)
            _f_out = vec()
            for l in range(self.velocity_set.q):
                _f_out[l] = _f_old[l] - _error_approx[l]

            write_population_to_global(fine, _f_out, i, j)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, fine, coarse):
        coarse_nodes_x = coarse.shape[1]
        coarse_nodes_y = coarse.shape[2]
        wp.launch(
            self.warp_kernel,
            inputs=[fine, coarse, coarse_nodes_x, coarse_nodes_y],
            dim=fine.shape[1:],
        )
