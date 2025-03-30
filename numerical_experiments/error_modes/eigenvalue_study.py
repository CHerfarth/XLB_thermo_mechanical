import sympy as sp
import xlb
from xlb.velocity_set import D2Q9
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy

#vars:
theta = 1/3
E = 0.085*2.5
nu = 1001.8
#K = E / (2 * (1 - nu))
#mu = E / (2 * (1 + nu))
K, mu, k, phi = sp.symbols('K mu k ph')

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)


I = sp.eye(8)

omega_11 = 1.0 / (mu / theta + 0.5)
omega_s = 1.0 / (2 * (1 / (1 + theta)) * K + 0.5)
omega_d = 1.0 / (2 * (1 / (1 - theta)) * mu + 0.5)
tau_12 = 0.5
tau_21 = 0.5
tau_f = 0.5
omega_12 = 1 / (tau_12 + 0.5)
omega_21 = 1 / (tau_21 + 0.5)
omega_f = 1 / (tau_f + 0.5)

omega = [0,0,omega_11, omega_s, omega_d, omega_12, omega_21, omega_f]
D = sp.diag(*omega)

# Create the transformation matrix
M = sp.zeros(8, 8)

# Fill in the matrix based on the given equations
M[0, 3-1] = 1.0
M[0, 6-1] = -1.0
M[0, 7-1] = 1.0
M[0, 4-1] = -1.0
M[0, 8-1] = -1.0
M[0, 5-1] = 1.0

M[1, 1-1] = 1.0
M[1, 2-1] = -1.0
M[1, 7-1] = 1.0
M[1, 4-1] = 1.0
M[1, 8-1] = -1.0
M[1, 5-1] = -1.0

M[2, 7-1] = 1.0
M[2, 4-1] = -1.0
M[2, 8-1] = 1.0
M[2, 5-1] = -1.0

M[3, 3-1] = 1.0
M[3, 1-1] = 1.0
M[3, 6-1] = 1.0
M[3, 2-1] = 1.0
M[3, 7-1] = 2.0
M[3, 4-1] = 2.0
M[3, 8-1] = 2.0
M[3, 5-1] = 2.0

M[4, 3-1] = 1.0
M[4, 1-1] = -1.0
M[4, 6-1] = 1.0
M[4, 2-1] = -1.0

M[5, 7-1] = 1.0
M[5, 4-1] = -1.0
M[5, 8-1] = -1.0
M[5, 5-1] = 1.0

M[6, 7-1] = 1.0
M[6, 4-1] = 1.0
M[6, 8-1] = -1.0
M[6, 5-1] = -1.0

M[7, 7-1] = 1.0
M[7, 4-1] = 1.0
M[7, 8-1] = 1.0
M[7, 5-1] = 1.0

# Compute the gamma factor and adjust M[7] (row 7)
tau_s = 2.0 * K/ (1.0 + theta)
gamma = (theta * tau_f) / ((1.0 + theta) * (tau_s - tau_f))

# Add gamma * row 3 to row 7
M[7, :] += gamma * M[3, :]

'''print(M)
print(M[:8,1:])
print(M[:8,1:].det())'''

M_inv = sp.zeros(8,8)
M_inv = M.inv()

# Create the matrix M_eq
M_eq = sp.zeros(8,8)
M_eq[0,0] = 1
M_eq[1,1] = 1
M_eq[5,1] = theta
M_eq[6,1] = theta



L_mat = M_inv*D*M_eq*M + M_inv*(I-D)*M

k_x = k*sp.cos(phi)
k_y = k*sp.sin(phi)

for i in range(velocity_set.q - 1):
    L_mat[i,:] *= sp.exp(-sp.I*(k_x * velocity_set.c[0, i+1] + k_y * velocity_set.c[1,i+1]))
'''
print("---------Starting calculation---------")
det = L.det(method="det_LU")
print("-----------Got determinant----------")
eqn=sp.Eq(det,0);
print("-----------Got characteristic equation------")
eigenvalues = sp.solve(eqn)
print("---------Got eigenvalues---------")
spectral_radius = max(abs(ev) for ev in eigenvalues)

dx = 1
dt = 0.001
for i in range(100):
    L = dx
    T = dt
    K_val = (E / (2*(1-nu)))*(T / (L*L))
    mu_val = (E / (2*(1+nu))) * (T/(L*L))
    evaluated_spectral_radius = spectral_radius.subs({mu: mu_val, K: K_val})
    print(evaluated_spectral_radius)'''


#alternative:
dx = 1
dt = 0.1
phi_val = 0
k_val = 1
#print(L_mat)
for i in range(100):
    L = dx
    T = dt
    K_val = (E / (2*(1-nu)))*(T / (L*L))
    mu_val = (E / (2*(1+nu))) * (T/(L*L))
    L_evaluated = L_mat.subs({mu: mu_val, K: K_val, phi: phi_val, k: k_val})
    eigenvalues = L_evaluated.eigenvals()
    spectral_radius = max(abs(ev) for ev in eigenvalues)
    print(spectral_radius)
    dx = dx*0.5






