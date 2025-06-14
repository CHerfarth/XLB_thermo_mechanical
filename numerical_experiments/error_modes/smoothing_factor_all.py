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
import argparse
import cmath


def is_normal_matrix(A, tol=1e-10):
    if A.shape[0] != A.shape[1]:
        return False  # Must be square
    A_star = A.conj().T
    return np.allclose(A @ A_star, A_star @ A, atol=tol)


parser = argparse.ArgumentParser("Smoothing Factor Study")
parser.add_argument("gamma", type=float)
args = parser.parse_args()
gamma_relax = args.gamma

# vars:
theta = 1 / 3

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
    if np.isclose(tau_s, tau_f):
        return I
    gamma_moments = (theta * tau_f) / ((1.0 + theta) * (tau_s - tau_f))

    # Add gamma * row 3 to row 7
    M[7, :] += np.float64(gamma_moments) * M[3, :]

    M_inv = np.linalg.inv(M)

    # Create the matrix M_eq
    M_eq = np.zeros(shape=(8, 8), dtype=np.complex128)
    M_eq[0, 0] = 1
    M_eq[1, 1] = 1
    M_eq[5, 0] = theta
    M_eq[6, 1] = theta

    # for relaxation
    gamma = args.gamma
    L_mat = gamma * (M_inv @ D @ M_eq @ M + M_inv @ (I - D) @ M)

    for i in range(velocity_set.q - 1):
        L_mat[i, :] *= cmath.exp(
            -1j * (phi_x * velocity_set.c[0, i + 1] + phi_y * velocity_set.c[1, i + 1])
        )

    L_mat += (1 - gamma) * I

    return L_mat


outer_iterations = 50
inner_iterations = 100
data_amplification = list()
data_difference = list()
data_smoothing_normal = list()
data_spectral_norms = list()

d_nu = 1 / outer_iterations
d_E = 1 / outer_iterations

for k in range(outer_iterations):
    nu = 0.01 + d_nu * k
    print("Nu: {}".format(nu))
    for l in range(outer_iterations):
        E = 0 + d_E * l
        print("E: {}".format(E))
        # cycle through all error modes
        smoothing_factors = list()
        spectral_radii = list()
        spectral_norms = list()
        phi_y_val = -np.pi
        for i in range(inner_iterations):
            dx = 1
            phi_x_val = -np.pi
            for j in range(inner_iterations):
                K_val = E / (2 * (1 - nu))
                mu_val = E / (2 * (1 + nu))
                L_evaluated = get_LB_matrix(
                    mu=mu_val, theta=theta, K=K_val, phi_x=phi_x_val, phi_y=phi_y_val
                )

                # check for normality
                # assert(is_normal_matrix(np.array(L_evaluated, dtype=np.complex128)))

                spectral_norm = np.linalg.norm(np.array(L_evaluated, dtype=np.complex128))
                eigenvalues = np.linalg.eig(np.array(L_evaluated, dtype=np.complex128)).eigenvalues
                spectral_radius = max(np.abs(ev) for ev in eigenvalues)
                spectral_norms.append(spectral_norm)
                if np.abs(phi_x_val) != 0.0 and np.abs(phi_y_val) != 0.0:
                    spectral_radii.append(spectral_radius)
                if np.abs(phi_x_val) >= 0.5 * np.pi and np.abs(phi_y_val) >= 0.5 * np.pi:
                    # print(spectral_radius)
                    smoothing_factors.append(spectral_radius)
                phi_x_val += (2 * np.pi) / inner_iterations
            phi_y_val += (2 * np.pi) / inner_iterations

        data_amplification.append((E, nu, np.max(smoothing_factors)))
        data_difference.append((E, nu, np.max(spectral_radii) - np.max(smoothing_factors)))
        data_smoothing_normal.append((E, nu, np.max(spectral_radii)))
        data_spectral_norms.append((E, nu, np.max(spectral_norms)))

        print("Max amplification factor: {}".format(np.max(smoothing_factors)))
        print("Max smoothing factor normal: {}".format(np.max(spectral_radii)))
    print("{} % complete".format((k + 1) * 100 / outer_iterations))


# ----------------------Plot amplification factors-------------------------
x = np.array([float(item[0]) for item in data_amplification])
y = np.array([float(item[1]) for item in data_amplification])
z = np.array([float(item[2]) for item in data_amplification])

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
ax.set_xlabel("E_scaled")
ax.set_ylabel("nu")
plt.title("Plot of Amplification Factor")

# Show the plot
plt.savefig("amplification_factors.png")


# ----------------------Plot spectral norms-------------------------
x = np.array([float(item[0]) for item in data_spectral_norms])
y = np.array([float(item[1]) for item in data_spectral_norms])
z = np.array([float(item[2]) for item in data_spectral_norms])

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
ax.set_xlabel("E_scaled")
ax.set_ylabel("nu")
plt.title("Plot of Spectral Norms")

# Show the plot
plt.savefig("spectral_norm.png")


# ----------------------------Plot difference to non-multigrid smoothing------------------

x = np.array([float(item[0]) for item in data_difference])
y = np.array([float(item[1]) for item in data_difference])
z = np.array([float(item[2]) for item in data_difference])

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
ax.set_xlabel("E_scaled")
ax.set_ylabel("nu")
plt.title("Plot of Difference")

# Show the plot
plt.savefig("difference.png")


# ----------------------------Plot non-multigrid smoothing stability------------------
indicator = lambda x: 10 if x > 1 else 0

x = np.array([float(item[0]) for item in data_smoothing_normal])
y = np.array([float(item[1]) for item in data_smoothing_normal])
z = np.array([indicator(float(item[2])) for item in data_smoothing_normal])

# Create a grid of points
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

# Interpolate the scattered data onto the grid
z_grid = griddata((x, y), z, (x_grid, y_grid))

# Create a 2D contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(x_grid, y_grid, z_grid, levels=1, cmap="viridis")


# Add color bar to the plot
plt.colorbar(contour)

# Set labels
ax.set_xlabel("E_scaled")
ax.set_ylabel("nu")
plt.title("Plot of Stability")

# Show the plot
plt.savefig("stability.png")

# ----------------------------Plot non-multigrid smoothing factors------------------

x = np.array([float(item[0]) for item in data_smoothing_normal])
y = np.array([float(item[1]) for item in data_smoothing_normal])
z = np.array([float(item[2]) for item in data_smoothing_normal])

# Create a grid of points
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

# Interpolate the scattered data onto the grid
z_grid = griddata((x, y), z, (x_grid, y_grid), method="cubic")

# Create a 2D contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap="viridis")


# Add color bar to the plot
plt.colorbar(contour)

# Set labels
ax.set_xlabel("E_scaled")
ax.set_ylabel("nu")
plt.title("Plot of Smoothing Factor normal LB")

# Show the plot
plt.savefig("normal_smoothing.png")
