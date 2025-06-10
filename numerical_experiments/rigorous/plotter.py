import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import argparse
from mpltools import annotation
import math


parser = argparse.ArgumentParser("plotter")
parser.add_argument("filename", type=str)
args = parser.parse_args()

data = pd.read_csv(args.filename, skiprows=0, sep=",", engine="python", dtype=np.float64)


def plot_convergence(nu):
    df = data[data["nu"] == nu]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = df["v1"]
    y = df["v2"]
    z = df["E"]
    c = df["converged"].map({0: "red", 1: "green", 2: "grey"})  # color by result

    ax.scatter(x, y, z, c=c)
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_zlabel("E")
    plt.title("Convergence for nu={}".format(nu))
    plt.savefig("nu_{}_convergence.png".format(nu))

def plot_efficiency(nu):
    df = data[data["nu"] == nu]
    df = df[df["converged"] == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = df["v1"]
    y = df["v2"]
    z = df["E"]
    c = -df["WU_per_iteration"]/np.log(df["rate"])
    sc = ax.scatter(x, y, z, c=c, cmap="viridis")

    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_zlabel("E")
    plt.title("Efficiency for nu={}".format(nu))
    plt.colorbar(sc, ax=ax, label="rate_of_convergence")
    plt.savefig("nu_{}_efficiency.png".format(nu))


for nu in data["nu"].unique():
    plot_convergence(nu)
    plot_efficiency(nu)

