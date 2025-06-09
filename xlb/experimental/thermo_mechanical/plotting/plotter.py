import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_single(data, x_label, y_label, title, name, color="blue", xlim=None, ylim=None):
    data = data.to_numpy()
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], color=color)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(name)


def plot_double(
    data1,
    data2,
    x_label,
    y_label,
    title,
    name,
    color1="blue",
    color2="red",
    label1=None,
    label2=None,
    xlim=None,
    ylim=None,
    scatter=False,
):
    data1 = data1.to_numpy()
    data2 = data2.to_numpy()
    fig, ax = plt.subplots()
    if scatter == False:
        ax.plot(data1[:, 0], data1[:, 1], "--", color=color1, label=label1)
        ax.plot(data2[:, 0], data2[:, 1], "--", color=color2, label=label2)
    else:
        ax.scatter(data1[:, 0], data1[:, 1], color=color1, label=label1)
        ax.scatter(data2[:, 0], data2[:, 1], color=color2, label=label2)
    ax.grid(True)
    plt.yscale("log")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    if label1 != None and label2 != None:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(name)


data = pd.read_csv("results.csv", skiprows=0, sep=",", engine="python", dtype=np.float64)
print(data.head())
plot_double(
    data.drop("Linf", axis=1),
    data.drop("L2", axis=1),
    "Timestep",
    "Error",
    "Error Norms over Time",
    "results.png",
    label1="L2 Norm",
    label2="L-Inf Norm",
)
