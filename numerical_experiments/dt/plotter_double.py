import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser("convergence_study")
parser.add_argument("results_csv",type=str)
parser.add_argument("dt",type=float)
parser.add_argument("output_file")
args = parser.parse_args()


data = pd.read_csv(args.results_csv, skiprows=0, sep=",", engine="python", dtype=np.float64)
print(data.head())
x_label = "Timestep Size"
y_label = "Timesteps needed to Convergence"
title = "Effect on timestep on interations"
dt = args.dt

data['time'] = data['timestep']


# Create subplots (2 plots side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot
ax1.plot(data['dt'], data['iterations'], "-", color='blue', label='Iterations')
ax1.set_title(title)
ax1.set_xlabel(x_label, labelpad=20, fontsize=12)
ax1.set_ylabel(y_label, labelpad=20, fontsize=12)
ax1.grid(True)
ax1.set_yscale("log")
ax1.legend(loc="upper right")

# Second plot (example plot, modify as needed)
ax2.plot(data['dt'], data['l2_disp'], "-", color='green', label='Time vs Dt')  # Change this to your desired column/plot
ax2.set_title("Time vs Timestep Size")
ax2.set_xlabel(x_label, labelpad=20, fontsize=12)
ax2.set_ylabel("Time", labelpad=20, fontsize=12)
ax2.grid(True)
ax2.legend(loc="upper right")


plt.tight_layout()
plt.savefig(args.output_file)
