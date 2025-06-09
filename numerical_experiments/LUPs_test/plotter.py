import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("plotter")
parser.add_argument("data", type=str)
args = parser.parse_args()

# Load CSV data
data = pd.read_csv(args.data)


# Group by dimension and compute mean and std
stats = data.groupby("dim")["MLUP/s"].agg(["mean", "std"]).reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(
    stats["dim"], stats["mean"], yerr=stats["std"], fmt="o-", capsize=5, label="Standard LB"
)

# Add labels and legend
plt.xlabel("Dimension")
plt.ylabel("MLUP/s")
plt.xscale("log", base=2)
plt.yscale("log")
plt.title("Lattice Updates vs Dimension")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig("speed.png")
