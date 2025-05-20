import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse
from mpltools import annotation


parser = argparse.ArgumentParser("plotter")
parser.add_argument("amplification_factor", type=float)
parser.add_argument("smoothing_steps_per_iteration", type=int)
parser.add_argument("smoothing_factor", type=float)
args = parser.parse_args()

amplification_factor = args.amplification_factor
smoothing_steps_per_iteration = args.smoothing_steps_per_iteration
smoothing_factor = args.smoothing_factor


multigrid_data = pd.read_csv("multigrid_results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(multigrid_data.head())
normal_data = pd.read_csv("normal_results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(normal_data.head())


# plot convergence residual per iteration for multigrid
title = "Residual over Iteration for Multigrid LB"
x_label = "Iteration"
y_label = "Residual"
fig, ax = plt.subplots()
ax.plot(multigrid_data["iteration"], multigrid_data["residual_norm"], "-", color="green", label="Residual")
ax.grid(True)
plt.yscale("log")
ax.set_title(title)
# plot expected speed of convergence
slope = amplification_factor**smoothing_steps_per_iteration
multigrid_data["slope_power"] = slope ** multigrid_data["iteration"]
ax.plot(multigrid_data["iteration"], multigrid_data["slope_power"], "--", color="black", label="Expected Speed of convergence")
ax.set_ylim((1e-11, 1e2))
# calculate actual speed of convergence
end_residual = multigrid_data["residual_norm"].min()
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("residual_mg.png")


# plot convergence residual per iteration for standard
title = "Residual over Iteration for Standard LB"
x_label = "Iteration"
y_label = "Residual"
fig, ax = plt.subplots()
ax.plot(normal_data["iteration"], normal_data["residual_norm"], "-", color="green", label="Residual")
ax.grid(True)
plt.yscale("log")
ax.set_title(title)
# plot expected speed of convergence
slope = smoothing_factor
normal_data["slope_power"] = slope ** (normal_data["iteration"])
ax.plot(normal_data["iteration"], normal_data["slope_power"], "--", color="black", label="Expected Speed of convergence")
ax.set_ylim((1e-11, 1e2))
# calculate actual speed of convergence
end_residual = normal_data["residual_norm"].min()
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("residual_standard.png")
