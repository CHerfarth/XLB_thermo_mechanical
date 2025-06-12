from xlb.operator import Operator
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
from xlb.compute_backend import ComputeBackend
import warp as wp


class Restriction(Operator):
    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
        with_boundary=False
    ):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        self.with_boundary = with_boundary

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
            f_out = f_a + f_b + f_c + f_d
            return f_out

        @wp.kernel
        def kernel_no_bc(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
        ):
            i, j, k = wp.tid()

            _f_a = read_local_population(fine, 2 * i, 2 * j)
            _f_b = read_local_population(fine, 2 * i + 1, 2 * j)
            _f_c = read_local_population(fine, 2 * i, 2 * j + 1)
            _f_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)

            _f_out = functional(f_a=_f_a, f_b=_f_b, f_c=_f_c, f_d=_f_d)

            write_population_to_global(coarse, _f_out, i, j)

        @wp.kernel
        def kernel_with_bc(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
            fine_boundary_array: wp.array4d(dtype=wp.int8),
        ):
            i, j, k = wp.tid()

            _f_a = read_local_population(fine, 2 * i, 2 * j)
            _f_b = read_local_population(fine, 2 * i + 1, 2 * j)
            _f_c = read_local_population(fine, 2 * i, 2 * j + 1)
            _f_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)

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
            
            if (domain_a and domain_b and domain_c and domain_d):
                _f_out = functional(f_a=_f_a, f_b=_f_b, f_c=_f_c, f_d=_f_d)
            elif domain_a:
                _f_out = functional(f_a=_f_a, f_b=_f_a, f_c=_f_a, f_d=_f_a)
            elif domain_b:
                _f_out = functional(f_a=_f_b, f_b=_f_b, f_c=_f_b, f_d=_f_b)
            elif domain_c:
                _f_out = functional(f_a=_f_c, f_b=_f_c, f_c=_f_c, f_d=_f_c)
            elif domain_d:
                _f_out = functional(f_a=_f_d, f_b=_f_d, f_c=_f_d, f_d=_f_d)
            else:
                _f_out = vec()
                for l in range(self.velocity_set.q):
                    _f_out[l] += self.compute_dtype(0)

            write_population_to_global(coarse, _f_out, i, j)

        return functional, (kernel_no_bc, kernel_with_bc)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, fine, coarse, fine_boundary_array=None):
        if fine_boundary_array is None:
            wp.launch(self.warp_kernel[0], inputs=[fine, coarse], dim=coarse.shape[1:])
        else:
            wp.launch(self.warp_kernel[1], inputs=[fine, coarse, fine_boundary_array], dim=coarse.shape[1:])

