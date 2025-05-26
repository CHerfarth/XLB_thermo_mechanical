import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import argparse
from mpltools import annotation


parser = argparse.ArgumentParser("plotter")
parser.add_argument("filename", type=str)
args = parser.parse_args()

data = pd.read_csv(args.filename, skiprows=0, sep=",", engine="python", dtype=np.float64)

def plot_for_nu(nu):
    df = data[data['nu'] == nu]
    print(df.head())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['v1']
    y = df['v2']
    z = df['E']
    c = df['converged'].map({0: 'red', 1: 'green', 2: 'grey'})  # color by result

    ax.scatter(x, y, z, c=c)
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    ax.set_zlabel('E')
    plt.title('Convergence for nu={}'.format(nu))
    plt.savefig('nu_{}.png'.format(nu))

for nu in data['nu'].unique():
    plot_for_nu(nu)

