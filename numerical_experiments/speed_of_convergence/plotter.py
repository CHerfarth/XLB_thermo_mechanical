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
args = parser.parse_args()

amplification_factor = args.amplification_factor
smoothing_steps_per_iteration = args.smoothing_steps_per_iteration


multigrid_data = pd.read_csv("multigrid_results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(multigrid_data.head())


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
slope = amplification_factor**4#**smoothing_steps_per_iteration
multigrid_data["slope_power"] = slope ** multigrid_data["iteration"]
ax.plot(multigrid_data["iteration"], multigrid_data["slope_power"], "--", color="black", label="Expected Speed of convergence")
ax.set_ylim((1e-14, 1))
# calculate actual speed of convergence
end_residual = multigrid_data["residual_norm"].min()
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("residual_over_iteration.png")
