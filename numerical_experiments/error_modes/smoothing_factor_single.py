import xlb
from xlb.velocity_set import D2Q9
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
import argparse
import cmath

parser = argparse.ArgumentParser("amplification_factor")
parser.add_argument("E", type=float)
parser.add_argument("nu", type=float)
args = parser.parse_args()
# vars:
theta = 1 / 3
E = args.E
nu = args.nu

# K = E / (2 * (1 - nu))
# mu = E / (2 * (1 + nu))

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(
    precision_policy=precision_policy, compute_backend=compute_backend
)

xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)


def get_LB_matrix(mu, theta, K, phi_x, phi_y):
    I = np.eye(8)

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
    D = np.diag(omega)

    # Create the transformation matrix
    M = np.zeros(shape=(8, 8), dtype=np.complex128)

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
    M[7, :] += np.float64(gamma) * M[3, :]

    M_inv = np.linalg.inv(M)

    # Create the matrix M_eq
    M_eq = np.zeros(shape=(8, 8), dtype=np.complex128)
    M_eq[0, 0] = 1
    M_eq[1, 1] = 1
    M_eq[5, 0] = theta
    M_eq[6, 1] = theta

    # for relaxation
    gamma = 0.8
    L_mat = gamma * (M_inv @ D @ M_eq @ M + M_inv @ (I - D) @ M)

    for i in range(velocity_set.q - 1):
        L_mat[i, :] *= cmath.exp(
            -1j * (phi_x * velocity_set.c[0, i + 1] + phi_y * velocity_set.c[1, i + 1])
        )

    L_mat += (1 - gamma) * I

    return L_mat


phi_y_val = -np.pi
iterations = 250
results = list()
for i in range(iterations):
    dx = 1
    phi_x_val = -np.pi
    for j in range(iterations):
        K_val = E / (2 * (1 - nu))
        mu_val = E / (2 * (1 + nu))
        L_evaluated = get_LB_matrix(
            mu=mu_val, theta=theta, K=K_val, phi_x=phi_x_val, phi_y=phi_y_val
        )
        eigenvalues = np.linalg.eig(L_evaluated).eigenvalues
        spectral_radius = max(np.abs(ev) for ev in eigenvalues)
        # spectral_radius = np.linalg.norm(np.array(L_evaluated, dtype=np.complex128), ord=2)
        results.append((phi_x_val, phi_y_val, spectral_radius))
        phi_x_val += (2 * np.pi) / iterations
    print("{} % complete".format((i + 1) * 100 / iterations))
    phi_y_val += (2 * np.pi) / iterations


x = np.array([float(item[0]) for item in results])
y = np.array([float(item[1]) for item in results])
z = np.array([float(item[2]) for item in results])

smoothing_factors = list()
for item in results:
    theta_1 = item[0]
    theta_2 = item[1]
    if np.abs(theta_1) >= 0.5 * np.pi or np.abs(theta_2) >= 0.5 * np.pi:
        smoothing_factors.append(np.abs(item[2]))
smoothing_factor = np.max(smoothing_factors)


# Create a grid of points
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

# Interpolate the scattered data onto the grid
z_grid = griddata((x, y), z, (x_grid, y_grid), method="cubic")

# Create a 2D contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(x_grid, y_grid, z_grid, levels=30, cmap="viridis")


# Add color bar to the plot
plt.colorbar(contour)

# Set labels
ax.set_xlabel("phi_x")
ax.set_ylabel("phi_y")
plt.title("Plot of Spectral Radius")

# Show the plot
plt.savefig("contour.png")

print("Smoothing Factor: {}".format(smoothing_factor))
