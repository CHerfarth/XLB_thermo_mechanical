import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("plotter")
parser.add_argument("data", type=str)
args = parser.parse_args()

# Load CSV data
data = pd.read_csv(args.data)

# Filter only converged entries
multigrid_data = data[data['multigrid_converged'] == 1]
standard_data = data[data['standard_converged'] == 1]

# Group by dimension and compute mean and std
multigrid_stats = multigrid_data.groupby('dim')['multigrid_time'].agg(['mean', 'std']).reset_index()
standard_stats = standard_data.groupby('dim')['standard_time'].agg(['mean', 'std']).reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(multigrid_stats['dim'], multigrid_stats['mean'], yerr=multigrid_stats['std'],
             fmt='o-', capsize=5, label='Multigrid Method')
plt.errorbar(standard_stats['dim'], standard_stats['mean'], yerr=standard_stats['std'],
             fmt='s-', capsize=5, label='Standard Method')

# Add labels and legend
plt.xlabel('Dimension')
plt.ylabel('Runtime (seconds)')
plt.xscale('log', base=2)
plt.yscale('log')
plt.title('Average Runtime vs Dimension (Only Converged Cases)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('runtimes.png')
