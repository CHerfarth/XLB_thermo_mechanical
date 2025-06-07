import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser("convergence_study")
parser.add_argument("results_csv", type=str)
parser.add_argument("output_file")
args = parser.parse_args()


data = pd.read_csv(args.results_csv, skiprows=0, sep=",", engine="python", dtype=np.float64)
print(data.head())
x_label = "Timestep Size"


# Create subplots (2 plots side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot
ax1.plot(data["dt"], data["iteration"], "-", color="blue", label="Iterations")
ax1.set_title("Timesteps to convergence")
ax1.set_xlabel(x_label, labelpad=20, fontsize=12)
ax1.set_ylabel("Timesteps needed to convergence", labelpad=20, fontsize=12)
ax1.grid(True)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.legend(loc="upper right")

# Second plot (example plot, modify as needed)
ax2.plot(
    data["dt"], data["l2_disp"], "-", color="green", label="L2 disp"
)  # Change this to your desired column/plot
ax2.plot(
    data["dt"], data["linf_disp"], "--", color="green", label=("Linf disp")
)  # Change this to your desired column/plot
ax2.plot(
    data["dt"], data["l2_stress"], "-", color="orange", label=("L2 stress")
)  # Change this to your desired column/plot
ax2.plot(
    data["dt"], data["linf_stress"], "--", color="orange", label=("Linf stress")
)  # Change this to your desired column/plot
ax2.set_title("Error")
ax2.set_xlabel(x_label, labelpad=20, fontsize=12)
ax2.set_ylabel("Error", labelpad=20, fontsize=12)
ax2.grid(True)
ax2.legend(loc="upper right")
ax2.set_yscale("log")
ax2.set_xscale("log")


plt.tight_layout()
plt.savefig(args.output_file)
