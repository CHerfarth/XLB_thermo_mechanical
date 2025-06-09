import sympy as sp
import xlb
from xlb.velocity_set import D2Q9
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
from xlb.experimental.thermo_mechanical.solid_stepper import SimulationParams
from xlb.experimental.thermo_mechanical.solid_stepper import SolidsStepper
from xlb.experimental.thermo_mechanical.solid_collision import SolidsCollision
from xlb.experimental.thermo_mechanical.kernel_provider import KernelProvider
import math
import warp as wp
import argparse


# vars:
theta = 1 / 3
# K = E / (2 * (1 - nu))
# mu = E / (2 * (1 + nu))
K, mu, k, phi = sp.symbols("K mu k ph")

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP64FP64
velocity_set = xlb.velocity_set.D2Q9(
    precision_policy=precision_policy, compute_backend=compute_backend
)

xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)
# initialize grid
nodes_x = 80
nodes_y = 80

# get discretization
length_x = 1
length_y = 1
dx = length_x / float(nodes_x)
dy = length_y / float(nodes_y)
assert math.isclose(dx, dy)
timesteps = 10000
dt = 0.001

# params
E = 0.085 * 2.5
nu = 0.8

solid_simulation = SimulationParams()
solid_simulation.set_all_parameters(E=E, nu=nu, dx=dx, dt=dt, L=dx, T=dt, kappa=1, theta=1.0 / 3.0)


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

omega = [0, 0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_f]
D = sp.diag(*omega)

# Create the transformation matrix
M = sp.zeros(8, 8)

# Fill in the matrix based on the given equations
M[0, 3 - 1] = 1.0
M[0, 6 - 1] = -1.0
M[0, 7 - 1] = 1.0
M[0, 4 - 1] = -1.0
M[0, 8 - 1] = -1.0
M[0, 5 - 1] = 1.0

M[1, 1 - 1] = 1.0
M[1, 2 - 1] = -1.0
M[1, 7 - 1] = 1.0
M[1, 4 - 1] = 1.0
M[1, 8 - 1] = -1.0
M[1, 5 - 1] = -1.0

M[2, 7 - 1] = 1.0
M[2, 4 - 1] = -1.0
M[2, 8 - 1] = 1.0
M[2, 5 - 1] = -1.0

M[3, 3 - 1] = 1.0
M[3, 1 - 1] = 1.0
M[3, 6 - 1] = 1.0
M[3, 2 - 1] = 1.0
M[3, 7 - 1] = 2.0
M[3, 4 - 1] = 2.0
M[3, 8 - 1] = 2.0
M[3, 5 - 1] = 2.0

M[4, 3 - 1] = 1.0
M[4, 1 - 1] = -1.0
M[4, 6 - 1] = 1.0
M[4, 2 - 1] = -1.0

M[5, 7 - 1] = 1.0
M[5, 4 - 1] = -1.0
M[5, 8 - 1] = -1.0
M[5, 5 - 1] = 1.0

M[6, 7 - 1] = 1.0
M[6, 4 - 1] = 1.0
M[6, 8 - 1] = -1.0
M[6, 5 - 1] = -1.0

M[7, 7 - 1] = 1.0
M[7, 4 - 1] = 1.0
M[7, 8 - 1] = 1.0
M[7, 5 - 1] = 1.0

# Compute the gamma factor and adjust M[7] (row 7)
tau_s = 2.0 * K / (1.0 + theta)
gamma = (theta * tau_f) / ((1.0 + theta) * (tau_s - tau_f))

# Add gamma * row 3 to row 7
M[7, :] += gamma * M[3, :]

M_inv = M.inv()

# Create the matrix M_eq
M_eq = sp.zeros(8, 8)
M_eq[0, 0] = 1
M_eq[1, 1] = 1
M_eq[5, 0] = theta
M_eq[6, 1] = theta


# test matrix
f = np.zeros(8)
for i in range(8):
    f[i] = (i + 1) ** 2
K_val = solid_simulation.K
mu_val = solid_simulation.mu
L_mat = M_inv * D * M_eq * M + M_inv * (I - D) * M
L_evaluated = L_mat.subs({mu: mu_val, K: K_val})
f_post = np.dot(np.array(L_evaluated).astype(np.float64), f)

print("------------------------------------")
print("With local mode:")
print(f_post)
print("------------------------------------")


mu = mu_val
K = K_val

# ----------calculate omega------------
omega_11 = 1.0 / (mu / theta + 0.5)
omega_s = 1.0 / (2 * (1 / (1 + theta)) * K + 0.5)
omega_d = 1.0 / (2 * (1 / (1 - theta)) * mu + 0.5)
tau_12 = 0.5
tau_21 = 0.5
tau_f = 0.5
omega_12 = 1 / (tau_12 + 0.5)
omega_21 = 1 / (tau_21 + 0.5)
omega_f = 1 / (tau_f + 0.5)
omega = KernelProvider().solid_vec(
    0.0, 0.0, omega_11, omega_s, omega_d, omega_12, omega_21, omega_f, 0.0
)

f = np.zeros(shape=(9, 1, 1, 1), dtype=np.float64)
force = np.zeros(shape=(2, 1, 1, 1), dtype=wp.float64)
for i in range(9):
    f[i, 0, 0, 0] = i**2
f_device = wp.from_numpy(f, dtype=wp.float64)
force_device = wp.from_numpy(force, dtype=wp.float64)
collision = SolidsCollision(omega)
wp.launch(
    collision.warp_kernel, inputs=[f_device, force_device, omega, theta], dim=f_device.shape[1:]
)
print("-------------------------------")
print("With collision:")
print(f_device.numpy()[1:, 0, 0, 0])
