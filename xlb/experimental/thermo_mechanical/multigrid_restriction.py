from xlb.operator import Operator
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
from xlb.compute_backend import ComputeBackend
import warp as wp

class Restriction(Operator):
    
    def __init__(
        self,
        velocity_set = None,
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
            f_out = (f_a + f_b + f_c + f_d)
            return f_out

            

        @wp.kernel
        def kernel(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
        ):
            i, j, k = wp.tid()

            _f_a = read_local_population(fine, 2 * i, 2 * j)
            _f_b = read_local_population(fine, 2 * i + 1, 2 * j)
            _f_c = read_local_population(fine, 2 * i, 2 * j + 1)
            _f_d = read_local_population(fine, 2 * i + 1, 2 * j + 1)

            _f_out = functional(f_a=_f_a, f_b=_f_b, f_c = _f_c, f_d=_f_d)

            write_population_to_global(coarse, _f_out, i, j)
        
        return functional, kernel
    
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, fine, coarse):
        wp.launch(self.warp_kernel, inputs=[fine, coarse], dim=coarse.shape[1:])

