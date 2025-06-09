import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("plotter")
parser.add_argument("data", type=str)
args = parser.parse_args()

# Load CSV data
data = pd.read_csv(args.data)



data_single = data[data["single_precision"]==1]
data_single = data_single.groupby("dim", group_keys=False).apply(lambda x: x.iloc[1:])
stats_single = data_single.groupby("dim")["MLUP/s"].agg(["mean", "std"]).reset_index()

data_double = data[data["single_precision"]==0]
data_double = data_double.groupby("dim", group_keys=False).apply(lambda x: x.iloc[1:])
stats_double = data_double.groupby("dim")["MLUP/s"].agg(["mean", "std"]).reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(
    stats_single["dim"]**2, stats_single["mean"], yerr=stats_single["std"], fmt="o-", color='blue', capsize=5, label="Single Precision"
)
plt.errorbar(
    stats_double["dim"]**2, stats_double["mean"], yerr=stats_double["std"], fmt="o-", color='red', capsize=5, label="Double Precision"
)

max_single = round(max(stats_single["mean"]))
max_double = round(max(stats_double["mean"]))
plt.axhline(y=max_single, color='blue', linestyle='--', linewidth=1.5, label='{} LUPS'.format(max_single))
plt.axhline(y=max_double, color='red', linestyle='--', linewidth=1.5, label='{} LUPS'.format(max_double))

#plt.axvline(x=2304*8, color='green', linestyle=':', linewidth=1.5, label='x=12288')
#plt.axvline(x=2304*64, color='green', linestyle='-', linewidth=1.5, label='x=12288')

# Add labels and legend
plt.xlabel("Grid points", fontsize=12)
plt.ylabel("MLUPS", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.title("Lattice Updates per Second vs Number of Grid Points")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig("speed.png")
plt.savefig("speed.eps")
