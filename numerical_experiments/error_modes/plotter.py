import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser("convergence_study")
parser.add_argument("results_csv",type=str)
parser.add_argument("dt",type=float)
parser.add_argument("k", type=int)
parser.add_argument("output_file")
args = parser.parse_args()


data = pd.read_csv(args.results_csv, skiprows=0, sep=",", engine="python", dtype=np.float64)
print(data.head())
x_label = "Timestep"
y_label = "Residual"
title = "Residual Norms over Time"
dt = args.dt

data['time'] = data['timestep']

cmap = plt.get_cmap('plasma')  # You can choose any colormap here

fig, ax = plt.subplots()
for i in range(args.k):
    ax.plot(data['timestep'], data[str(i)+'_residual']/data[str(i)+'_residual'][0], "-", color=cmap(i / args.k), label=str(i)+' Residual')
##ax.plot(data['time'], data['linf_disp'], "--", color='blue', label='Linf disp')
#ax.plot(data['time'], data['l2_stress'], "-", color='green', label='L2 stress')
#ax.plot(data['time'], data['linf_stress'], "--", color='green', label='Linf stress')


ax.grid(True)
plt.yscale("log")
ax.set_title(title)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)
plt.legend(loc="upper right")
plt.ylim(bottom=1e-7) #machine precision for float32s
plt.tight_layout()
plt.savefig(args.output_file)
