from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
from typing import Any


# Mapping:
#    i  j   |   q
#    1  0   |   0
#    0  1   |   1
#   -1  0   |   2
#    0 -1   |   3
#    1  1   |   4
#   -1  1   |   5
#   -1 -1   |   6
#    1 -1   |   7
#    0  0   |   8 (ignored)

precision_policy = PrecisionPolicy.FP32FP32
f_vec = wp.vec(9, dtype=precision_policy.compute_precision.wp_dtype)

@wp.func
def calc_moments(f: f_vec):
    m = f_vec()
    #Todo: find better way to do this!
    m[0] = f[0] - f[2] + f[4] - f[5] - f[6] + f[7]
    m[1] = f[1] - f[3] + f[4] + f[5] - f[6] - f[7]
    m[2] = f[4] - f[5] + f[6] - f[7]
    m[3] = f[0] + f[1] + f[2] + f[3] + 2.*f[4] + 2.*f[5] + 2.*f[6] + 2.*f[7]
    m[4] = f[0] - f[1] + f[2] - f[3]
    m[5] = f[4] - f[5] - f[6] + f[7]
    m[6] = f[4] + f[5] - f[6] - f[7]
    m[7] = f[4] + f[5] + f[6] + f[7]
    m[8] = 0. 
    return m

@wp.func
def calc_populations(m: f_vec):
    f = f_vec()
    #Todo: find better way to do this!
    f[0] = 2.*m[0] + m[3] + m[4] - 2.*m[5] - 2.*m[7]
    f[1] = 2.*m[1] + m[3] - m[4] - 2.*m[6] - 2.*m[7]
    f[2] = -2.*m[0] + m[3] + m[4] + 2.*m[5] - 2.*m[7]
    f[3] = -2.*m[1] + m[3] - m[4] + 2.*m[6] - 2.*m[7]
    f[4] = m[2] + m[5] + m[6] + m[7]
    f[5] = -m[2] - m[5] + m[6] + m[7]
    f[6] = m[2] - m[5] - m[6] + m[7]
    f[7] = -m[2] + m[5] - m[6] + m[7]
    f[8] = 0. 
    return f

@wp.func
def read_local(f: wp.array4d(dtype=Any), dim: wp.int32, x: wp.int32, y: wp.int32):
    f_local =  f_vec()
    for i in range(dim):
       f_local[i] = f[i, x, y, 0]
    return f_local

@wp.func
def write_global(f: wp.array4d(dtype=Any), f_local: f_vec, dim: wp.int32, x: wp.int32, y: wp.int32):
    for i in range(dim):
       f[i, x, y, 0] = f_local[i]

@wp.func
def calc_equilibrium(m: f_vec, theta: Any):
    m_eq = f_vec()
    m_eq[0] = m[0]
    m_eq[1] = m[1]
    m_eq[2] = 0.
    m_eq[3] = 0.
    m_eq[4] = 0.
    m_eq[5] = theta * m[0]
    m_eq[6] = theta * m[1]
    m_eq[7] = 0.
    m_eq[8] = 0.
    return m_eq


@wp.kernel
def collide(f: wp.array4d(dtype=Any), force: wp.array4d(dtype=Any), displacement: wp.array4d(dtype=Any), omega: f_vec, theta: Any):
    i, j, k = wp.tid() #for 2d, k will equal 1

    #calculate moments
    f_local = read_local(f, 9, i, j)
    m = calc_moments(f_local)

    #apply half-forcing and get displacement
    m[0] += 0.5*force[0, i, j, 0]
    m[1] += 0.5*force[1, i, j, 0]
    displacement[0,i,j, 0] = m[0]
    displacement[1,i,j, 0] = m[1]

    m_eq = calc_equilibrium(m, theta)

    #get post-collision populations
    for l in range(m._length_):
        m[l] = omega[l]*m_eq[l] + (1.-omega[l])*m[l]
    
    #half-forcing
    m[0] += 0.5*force[0, i, j, 0]
    m[1] += 0.5*force[1, i, j, 0]

    #get populations and write back to global
    f_local = calc_populations(m)
    write_global(f, f_local, 9, i, j)




if __name__ == "__main__":
    #set dimensions of domain
    domain_x = 3 #for now we work on square
    domain_y = 3

    #total time
    total_time = 2

    #set shape of grid
    nodes_x = 3
    nodes_y = 3
    timesteps = 40
    
    #calculate dx, dt
    dx = domain_x / nodes_x
    dy = domain_y / nodes_y
    dt = total_time / timesteps
    assert dx == dy, "Wrong spacial steps in directions x and y do not match"

    #init xlb stuff
    compute_backend = ComputeBackend.WARP
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=compute_backend)
    
    #initialise grid (should probably move this to Stepper class for full implementation later)
    grid = grid_factory((nodes_x, nodes_y), compute_backend=compute_backend)
    #vector_size = 3 #f_pre, f_eq, f_post
    f = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    force = grid.create_field(cardinality=2, dtype=precision_policy.store_precision)
    displacement = grid.create_field(cardinality=2, dtype=precision_policy.store_precision)
    
    
    #print startup info
    print("Initialized grid with dimensions     {}x{}".format(nodes_x, nodes_y))

    #-----------define variables-------------
    E = 0.085*2.5
    nu = 0.8
    mu = E/(2*(1+nu))
    lamb =  E/(2*(1-nu)) - mu
    K = lamb + mu
    theta = 1/3 #check this!!

    #-----------make dimensionless----------
    L = dx
    T = dt
    kappa = 1
    mu_scaled = mu * T / (L*L*kappa)
    lamb_scaled = lamb*T/(L*L*kappa)

    #calculate omega
    omega_11 = 1 / (mu_scaled / theta + 0.5)
    omega_d = 1 / (2 * mu_scaled / (1 - theta) + 0.5)
    omega_s = 1 / (2 * (mu_scaled + lamb_scaled) / (1 + theta) + 0.5)
    tau_11 = 1 / omega_11 - 0.5
    tau_s = 1 / omega_d - 0.5
    tau_p = 1 / omega_s - 0.5
    tau_12 = 0.5
    tau_21 = tau_12
    tau_22 = 0.5    #ToDo: Check these!!
    omega_12 = 1 / (tau_12 + 0.5)
    omega_21 = 1 / (tau_21 + 0.5)
    omega_22 = 1 / (tau_22 + 0.5)
    omega = f_vec(0., 0., omega_11, omega_s, omega_d, omega_12, omega_21, omega_22, 0.)




    #----------define foce load---------------
    #b_x = lambda x, y: (mu-K)*(cos(x))
    #b_y = lambda x, y: (mu-K)*(cos(y))
    
    #----------define exact solution-----------
    #exact_u = lambda x, y: cos(x)
    #exact_v = lambda x, y: cos(y)
    
    wp.launch(collide, inputs=[f, force, displacement, omega, theta], dim = f.shape[1:])

