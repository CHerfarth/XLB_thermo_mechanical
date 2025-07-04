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
parser.add_argument("gamma", type=float)
parser.add_argument("pre_smoothing_steps", type=int)
parser.add_argument("post_smoothing_steps", type=int)
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


def get_smoothing_matrix(mu, theta, K, phi_x, phi_y, smoothing_param):
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
    gamma = smoothing_param 
    L_mat = gamma * (M_inv @ D @ M_eq @ M + M_inv @ (I - D) @ M)

    for i in range(velocity_set.q - 1):
        L_mat[i, :] *= cmath.exp(
            -1j * (phi_x * velocity_set.c[0, i + 1] + phi_y * velocity_set.c[1, i + 1])
        )

    L_mat += (1 - gamma) * I

    return L_mat

def get_2h_smoothing_matrix(mu, theta, K, phi_x, phi_y, smoothing_param):
    L_mat = np.zeros(shape=(32,32), dtype=np.complex128) 
    # 0 0
    L_mat[0:8, 0:8] = get_smoothing_matrix(mu, theta, K, phi_x, phi_y, smoothing_param)
    # 1 1
    new_phi_x = phi_x - np.sign(phi_x)*np.pi
    new_phi_y = phi_y - np.sign(phi_y)*np.pi
    L_mat[8:16,8:16] = get_smoothing_matrix(mu, theta, K, new_phi_x, new_phi_y, smoothing_param)
    # 1 0
    L_mat[16:24, 16:24] = get_smoothing_matrix(mu, theta, K, new_phi_x, phi_y, smoothing_param)
    # 0 1
    L_mat[24:32, 24:32] = get_smoothing_matrix(mu, theta, K, phi_x, new_phi_y, smoothing_param)

    return L_mat

def get_restriction_matrix(mu, theta, K, phi_x, phi_y):
    R_mat = np.zeros(shape=(8, 32), dtype=np.complex128)

    #0 0
    val = 0.25*(1+np.cos(phi_x))*(1+np.cos(phi_y))
    R_mat[0:8, 0:8] = np.eye(8)*val
    # 1 1
    new_phi_x = phi_x - np.sign(phi_x)*np.pi
    new_phi_y = phi_y - np.sign(phi_y)*np.pi
    val = 0.25*(1+np.cos(new_phi_x))*(1+np.cos(new_phi_y))
    R_mat[0:8,8:16] = np.eye(8)*val
    # 1 0
    val = 0.25*(1+np.cos(new_phi_x))*(1+np.cos(phi_y))
    R_mat[0:8,16:24] = np.eye(8)*val
    # 0 1
    val = 0.25*(1+np.cos(phi_x))*(1+np.cos(new_phi_y))
    R_mat[0:8,24:32] = np.eye(8)*val

    return R_mat

def get_prolongation_matrix(mu, theta, K, phi_x, phi_y):
    P_mat = np.zeros(shape=(32,8), dtype=np.complex128)

    #0 0
    val = (1+np.cos(phi_x))*(1+np.cos(phi_y))
    P_mat[0:8, 0:8] = np.eye(8)*val
    # 1 1
    new_phi_x = phi_x - np.sign(phi_x)*np.pi
    new_phi_y = phi_y - np.sign(phi_y)*np.pi
    val = (1+np.cos(new_phi_x))*(1+np.cos(new_phi_y))
    P_mat[8:16,0:8] = np.eye(8)*val
    # 1 0
    val = (1+np.cos(new_phi_x))*(1+np.cos(phi_y))
    P_mat[16:24,0:8] = np.eye(8)*val
    # 0 1
    val = (1+np.cos(phi_x))*(1+np.cos(new_phi_y))
    P_mat[24:32,0:8] = np.eye(8)*val

    return P_mat

def get_L_coarse(mu, theta, K, phi_x, phi_y, factor=1):
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

    L_mat = (M_inv @ D @ M_eq @ M + M_inv @ (I - D) @ M)

    for i in range(velocity_set.q - 1):
        L_mat[i, :] *= cmath.exp(
            -1j * (factor*phi_x * velocity_set.c[0, i + 1] + factor*phi_y * velocity_set.c[1, i + 1])
        )
    
    L_mat = L_mat - I
    return L_mat

def get_L_fine(mu, theta, K, phi_x, phi_y):
    L_mat = np.zeros(shape=(32,32), dtype=np.complex128) 
    # 0 0
    L_mat[0:8, 0:8] = get_L_coarse(mu, theta, K, phi_x, phi_y)
    # 1 1
    new_phi_x = phi_x - np.sign(phi_x)*np.pi
    new_phi_y = phi_y - np.sign(phi_y)*np.pi
    L_mat[8:16,8:16] = get_L_coarse(mu, theta, K, new_phi_x, new_phi_y)
    # 1 0
    L_mat[16:24, 16:24] = get_L_coarse(mu, theta, K, new_phi_x, phi_y)
    # 0 1
    L_mat[24:32, 24:32] = get_L_coarse(mu, theta, K, phi_x, new_phi_y)

    return L_mat

iterations=200
K_val = E / (2 * (1 - nu))
mu_val = E / (2 * (1 + nu))
theta=1./3.
results = list()

phi_x = -np.pi/2
for i in range(iterations):
    phi_y = -np.pi/2
    for j in range(iterations):
        S = get_2h_smoothing_matrix(mu=mu_val, theta=theta, K=K_val, phi_x=phi_x, phi_y=phi_y, smoothing_param=args.gamma)
        pre_S = np.eye(32)
        for k in range(args.pre_smoothing_steps):
            pre_S = pre_S @ S
        post_S = np.eye(32)
        for k in range(args.post_smoothing_steps):
            post_S = post_S @ S
        L_fine = get_L_fine(mu=mu_val, theta=theta, K=K_val, phi_x=phi_x, phi_y=phi_y)
        L_coarse = get_L_coarse(mu=mu_val, theta=theta, K=K_val, phi_x=phi_x, phi_y=phi_y, factor=2)
        R = get_restriction_matrix(mu=mu_val, theta=theta, K=K_val, phi_x=phi_x, phi_y=phi_y)
        P = get_prolongation_matrix(mu=mu_val, theta=theta, K=K_val, phi_x=phi_x, phi_y=phi_y)

        M = post_S @ (np.eye(32) - P @ np.linalg.inv(L_coarse) @ R @ L_fine) @ pre_S

        eigenvalues = np.linalg.eig(M).eigenvalues
        spectral_radius = max(np.abs(ev) for ev in eigenvalues)

        if not (np.isclose(phi_x, 0) and np.isclose(phi_y, 0)):
            results.append((phi_x, phi_y, spectral_radius))
        phi_y += np.pi / (iterations-2)

    print("{} % complete".format((i + 1) * 100 / iterations))
    phi_x += np.pi / (iterations-2)

print(pre_S)
print(post_S)


x_grid, y_grid = np.meshgrid(np.linspace(-0.5*np.pi, 0.5*np.pi, 100), np.linspace(-0.5*np.pi, 0.5*np.pi, 100))
x = np.array([float(item[0]) for item in results])
y = np.array([float(item[1]) for item in results])
z = np.array([float(item[2]) for item in results])

# Print the 5 largest values of z with their respective x, y pairs
indices = np.argsort(z)[-5:][::-1]
for idx in indices:
    print(f"x: {x[idx]}, y: {y[idx]}, z: {z[idx]}")

print("Max Spectral Radius:", max(z))
# Interpolate the scattered data onto the grid
z_grid = griddata((x, y), z, (x_grid, y_grid), method="cubic")

# Create a 2D contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(x_grid, y_grid, z_grid, levels=100, cmap="viridis")

# Add color bar to the plot
plt.colorbar(contour)


def pi_formatter(x, pos):
    frac = x / np.pi
    if np.isclose(frac, 0):
        return r"$0$"
    elif np.isclose(frac, 1):
        return r"$\pi$"
    elif np.isclose(frac, -1):
        return r"$-\pi$"
    else:
        return r"${0}\pi$".format(int(frac) if frac == int(frac) else "{0:g}".format(frac))


ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

# Set labels
x_label = r"$\phi_1$"
y_label = r"$\phi_2$"
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.title("Amplification Factor")

# Show the plot
plt.tight_layout()
plt.savefig("contour_E_{}_nu_{}.pdf".format(args.E, args.nu))
