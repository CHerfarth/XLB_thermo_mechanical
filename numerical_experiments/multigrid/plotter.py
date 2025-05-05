import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse
from mpltools import annotation



multigrid_data = pd.read_csv("multigrid_results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(multigrid_data.head())
normal_data = pd.read_csv("normal_results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(normal_data.head())
max_wu_normal = normal_data["wu"].max()
print(max_wu_normal)

#plot convergence of residuals
title = "Residuals over Work Units (WU)"
x_label = "WU"
y_label = "Residual"
fig, ax = plt.subplots()
ax.plot(multigrid_data['wu'], multigrid_data['residual_norm'], "-", color='blue', label='Residual Multigrid LB')
ax.plot(normal_data['wu'], normal_data['residual_norm'], "-", color='green', label='Residual Standard LB')
ax.grid(True)
ax.set_xlim((0,max_wu_normal))
plt.yscale("log")
ax.set_title(title)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("residual_over_time.png")


#plot convergence of errors
title = "Error over Work Units (WU)"
x_label = "WU"
y_label = "Error"
fig, ax = plt.subplots()
ax.plot(multigrid_data['wu'], multigrid_data['l2_disp'], "-", color='blue', label='L2 disp Multigrid')
ax.plot(multigrid_data['wu'], multigrid_data['l2_stress'], "--", color='blue', label='L2 stress Multigrid')
ax.plot(normal_data['wu'], normal_data['l2_disp'], "-", color='green', label='L2 disp Standard')
ax.plot(normal_data['wu'], normal_data['l2_stress'], "--", color='green', label='L2 stress Standard')
ax.grid(True)
ax.set_xlim((0,max_wu_normal))
plt.yscale("log")
ax.set_title(title)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("error_over_time.png")


#plot all data for multigrid
title = "Error/Residual over Work Unit for Multigrid LB"
x_label = "WU"
y_label = "Residual/Error"
fig, ax = plt.subplots()
ax.plot(multigrid_data['wu'], multigrid_data['residual_norm'], "-", color='red', label='Residual')
ax.plot(multigrid_data['wu'], multigrid_data['l2_disp'], "-", color='blue', label='L2 disp')
ax.plot(multigrid_data['wu'], multigrid_data['linf_disp'], "--", color='blue', label='Linf disp')
ax.plot(multigrid_data['wu'], multigrid_data['l2_stress'], "-", color='green', label='L2 stress')
ax.plot(multigrid_data['wu'], multigrid_data['linf_stress'], "--", color='green', label='Linf stress')
ax.grid(True)
ax.set_xlim((0,max_wu_normal))
plt.yscale("log")
ax.set_title(title)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("multigrid_plot.png")


#plot all data for normal
title = "Error/Residual over Work Unit for Standard LB"
x_label = "WU"
y_label = "Residual/Error"
fig, ax = plt.subplots()
ax.plot(normal_data['wu'], normal_data['residual_norm'], "-", color='red', label='Residual')
ax.plot(normal_data['wu'], normal_data['l2_disp'], "-", color='blue', label='L2 disp')
ax.plot(normal_data['wu'], normal_data['linf_disp'], "--", color='blue', label='Linf disp')
ax.plot(normal_data['wu'], normal_data['l2_stress'], "-", color='green', label='L2 stress')
ax.plot(normal_data['wu'], normal_data['linf_stress'], "--", color='green', label='Linf stress')
ax.grid(True)
ax.set_xlim((0,max_wu_normal))
plt.yscale("log")
ax.set_title(title)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("normal_plot.png")


#plot convergence residual per iteration for multigrid
title = "Residual over Iteration for Multigrid LB"
x_label = "Iteration"
y_label = "Residual"
fig, ax = plt.subplots()
ax.plot(multigrid_data['iteration'], multigrid_data['residual_norm'], "-", color='green', label='Residual')
ax.grid(True)
plt.yscale("log")
ax.set_title(title)
#plot expected speed of convergence
slope = 0.714**4
multigrid_data['slope_power'] = slope ** multigrid_data['iteration']
ax.plot(multigrid_data['iteration'], multigrid_data['slope_power'], '--', color='black', label="Expected Speed of convergence = 0.714**4")
slope=0.93
multigrid_data['slope_power'] = slope ** multigrid_data['iteration']
#ax.plot(multigrid_data['iteration'], multigrid_data['slope_power'], '--', color='green', label="Speed of convergence = {}".format(slope))
print(slope)
ax.set_xlim((0, 300))
ax.set_ylim((1e-6,1))
#calculate actual speed of convergence
end_residual = multigrid_data["residual_norm"].min()
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("residual_over_iteration.png")

