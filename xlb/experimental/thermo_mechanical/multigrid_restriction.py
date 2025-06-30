from xlb.operator import Operator
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
from xlb.compute_backend import ComputeBackend
import warp as wp


class Restriction(Operator):
    def __init__(
        self, velocity_set=None, precision_policy=None, compute_backend=None, with_boundary=False
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
        zero_vec = kernel_provider.zero_vec

        @wp.func
        def functional(center: vec, up: vec, down: vec, left: vec, right: vec, dia_1: vec, dia_2: vec, dia_3: vec, dia_4: vec):
            f_out = self.compute_dtype(1./4.)*(
                self.compute_dtype(4)*center
                + self.compute_dtype(2)*(up + down + left + right)
                + self.compute_dtype(1)*(dia_1 + dia_2 + dia_3 + dia_4)
            )
            return f_out

        @wp.kernel
        def kernel_no_bc(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
            fine_nodes_x: wp.int32,
            fine_nodes_y: wp.int32,
        ):
            i, j, k = wp.tid()

            _center = read_local_population(fine, 2*i, 2*j)
            _up = read_local_population(fine, 2*i, wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _down = read_local_population(fine, 2*i, wp.mod(2*j-1+fine_nodes_y, fine_nodes_y))
            _right = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), 2*j)
            _left = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), 2*j)

            _dia_1 = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _dia_2 = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j-1+fine_nodes_y, fine_nodes_y))
            _dia_3 = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _dia_4 = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))

            _f_out = functional(center=_center, up=_up, down=_down, left=_left, right=_right, dia_1=_dia_1, dia_2=_dia_2, dia_3=_dia_3, dia_4=_dia_4)

            write_population_to_global(coarse, _f_out, i, j)

        @wp.kernel
        def kernel_with_bc(
            fine: wp.array4d(dtype=self.store_dtype),
            coarse: wp.array4d(dtype=self.store_dtype),
            fine_nodes_x: wp.int32,
            fine_nodes_y: wp.int32,
            fine_boundary_array: wp.array4d(dtype=wp.int8),
        ):
            i, j, k = wp.tid()

            _center = read_local_population(fine, 2*i, 2*j)
            _up = read_local_population(fine, 2*i, wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _down = read_local_population(fine, 2*i, wp.mod(2*j-1+fine_nodes_y, fine_nodes_y))
            _right = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), 2*j)
            _left = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), 2*j)

            _dia_1 = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _dia_2 = read_local_population(fine, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j-1+fine_nodes_y, fine_nodes_y))
            _dia_3 = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))
            _dia_4 = read_local_population(fine, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y))

            if fine_boundary_array[0, 2*i, 2*j, 0] == wp.int8(0):
                _center = zero_vec()
            if fine_boundary_array[0, 2*i, wp.mod(2*j+1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _up = zero_vec()
            if fine_boundary_array[0, 2*i, wp.mod(2*j-1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _down = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), 2*j, 0] == wp.int8(0):
                _right = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), 2*j, 0] == wp.int8(0):
                _left = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _dia_1 = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i+1+fine_nodes_x, fine_nodes_x), wp.mod(2*j-1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _dia_2 = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j+1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _dia_3 = zero_vec()
            if fine_boundary_array[0, wp.mod(2*i-1+fine_nodes_x, fine_nodes_x), wp.mod(2*j-1+fine_nodes_y, fine_nodes_y), 0] == wp.int8(0):
                _dia_4 = zero_vec()

            _f_out = functional(center=_center, up=_up, down=_down, left=_left, right=_right, dia_1=_dia_1, dia_2=_dia_2, dia_3=_dia_3, dia_4=_dia_4)

            write_population_to_global(coarse, _f_out, i, j)

        return functional, (kernel_no_bc, kernel_with_bc)

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, fine, coarse, fine_nodes_x, fine_nodes_y, fine_boundary_array=None):
        if fine_boundary_array is None:
            wp.launch(self.warp_kernel[0], inputs=[fine, coarse, fine_nodes_x, fine_nodes_y], dim=coarse.shape[1:])
        else:
            wp.launch(
                self.warp_kernel[1],
                inputs=[fine, coarse, fine_nodes_x, fine_nodes_y, fine_boundary_array],
                dim=coarse.shape[1:],
            )
