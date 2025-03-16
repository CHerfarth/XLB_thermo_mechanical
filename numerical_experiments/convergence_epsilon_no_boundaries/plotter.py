import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse

def draw_loglog_slope(fig, ax, origin, width_inches, slope, color, inverted=False, polygon_kwargs=None, label=True, labelcolor=None, label_kwargs=None, zorder=None):
    """
    This function draws slopes or "convergence triangles" into loglog plots.

    @param fig: The figure
    @param ax: The axes object to draw to
    @param origin: The 2D origin (usually lower-left corner) coordinate of the triangle
    @param width_inches: The width in inches of the triangle
    @param slope: The slope of the triangle, i.e. order of convergence
    @param inverted: Whether to mirror the triangle around the origin, i.e. whether 
        it indicates the slope towards the lower left instead of upper right (defaults to false)
    @param color: The color of the of the triangle edges (defaults to default color)
    @param polygon_kwargs: Additional kwargs to the Polygon draw call that creates the slope
    @param label: Whether to enable labeling the slope (defaults to true)
    @param labelcolor: The color of the slope labels (defaults to the edge color)
    @param label_kwargs: Additional kwargs to the Annotation draw call that creates the labels
    @param zorder: The z-order value of the triangle and labels, defaults to a high value
    """

    if polygon_kwargs is None:
        polygon_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}

    if color is not None:
        polygon_kwargs["color"] = color
    if "linewidth" not in polygon_kwargs:
        polygon_kwargs["linewidth"] = 0.75 * matplotlib.rcParams["lines.linewidth"]
    if labelcolor is not None:
        label_kwargs["color"] = labelcolor
    if "color" not in label_kwargs:
        label_kwargs["color"] = polygon_kwargs["color"]
    if "fontsize" not in label_kwargs:
        label_kwargs["fontsize"] = 0.75 * matplotlib.rcParams["font.size"]

    if inverted:
        width_inches = -width_inches
    if zorder is None:
        zorder = 10

    # For more information on coordinate transformations in Matplotlib see
    # https://matplotlib.org/3.1.1/tutorials/advanced/transforms_tutorial.html

    # Convert the origin into figure coordinates in inches
    origin_disp = ax.transData.transform(origin)
    origin_dpi = fig.dpi_scale_trans.inverted().transform(origin_disp)

    # Obtain the bottom-right corner in data coordinates
    corner_dpi = origin_dpi + width_inches * np.array([1.0, 0.0])
    corner_disp = fig.dpi_scale_trans.transform(corner_dpi)
    corner = ax.transData.inverted().transform(corner_disp)

    (x1, y1) = (origin[0], origin[1])
    x2 = corner[0]

    # The width of the triangle in data coordinates
    width = x2 - x1
    # Compute offset of the slope
    log_offset = y1 / (x1 ** slope)

    y2 = log_offset * ((x1 + width) ** slope)
    height = y2 - y1

    # The vertices of the slope
    a = origin
    b = corner
    c = [x2, y2]

    # Draw the slope triangle
    X = np.array([a, b, c])
    triangle = plt.Polygon(X[:3,:], fill=False, zorder=zorder, **polygon_kwargs)
    ax.add_patch(triangle)

    # Convert vertices into display space
    a_disp = ax.transData.transform(a)
    b_disp = ax.transData.transform(b)
    c_disp = ax.transData.transform(c)

    # Figure out the center of the triangle sides in display space
    bottom_center_disp = a_disp + 0.5 * (b_disp - a_disp)
    bottom_center = ax.transData.inverted().transform(bottom_center_disp)

    right_center_disp = b_disp + 0.5 * (c_disp - b_disp)
    right_center = ax.transData.inverted().transform(right_center_disp)

    # Label alignment depending on inversion parameter
    va_xlabel = "top" if not inverted else "bottom"
    ha_ylabel = "left" if not inverted else "right"

    # Label offset depending on inversion parameter
    offset_xlabel = [0.0, -0.33 * label_kwargs["fontsize"]] if not inverted else [0.0, 0.33 * label_kwargs["fontsize"]]
    offset_ylabel = [0.33 * label_kwargs["fontsize"], 0.0] if not inverted else [-0.33 * label_kwargs["fontsize"], 0.0]

    # Draw the slope labels
    ax.annotate("$1$", bottom_center, xytext=offset_xlabel, textcoords='offset points', ha="center", va=va_xlabel, zorder=zorder, **label_kwargs)
    ax.annotate(f"${slope}$", right_center, xytext=offset_ylabel, textcoords='offset points', ha=ha_ylabel, va="center", zorder=zorder, **label_kwargs)




def plot_single(data, x_label, y_label, title, name, color="blue", xlim=None, ylim=None):
    data = data.to_numpy()
    print(data)
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], color=color)
    plt.yscale('log')
    plt.xscale('log')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    draw_loglog_slope(fig, ax, (0.6,0.6), 1, 2, "black", inverted=True)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(name)


def plot_double(
    data1, data2, x_label, y_label, title, name, color1="blue", color2="red", label1=None, label2=None, xlim=None, ylim=None, scatter=False
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
    plt.yscale('log')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    if label1 != None and label2 != None:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(name)


#get command line arguments
parser = argparse.ArgumentParser("plot_convergence")
parser.add_argument("file", type=str)
args = parser.parse_args()

data = pd.read_csv(args.file, skiprows=0, sep=",", engine="python", dtype=np.float64)
print(data.head())
plot_single(data, "Epsilon", "L2 Norm of Error", "Convergence", "convergence.png")

x_label = "Epsilon"
y_label = "Error"
title = "Convergence"
name = "convergence.png"

#data = data.to_numpy()
fig, ax = plt.subplots()

#plot dispersement error
ax.plot(data['epsilon'], data['error_L2_disp'], "-ob",label='L2 disp')
ax.plot(data['epsilon'], data['error_Linf_disp'], "--ob", label='Linf disp')

#set scales, grid, title
plt.yscale('log')
plt.xscale('log')
ax.grid(True)
ax.set_title(title)

#plot convergence triangle
draw_loglog_slope(fig, ax, (0.6,0.6), 1, 2, "black", inverted=True)
plt.xlabel(x_label, labelpad=20, fontsize=12)
plt.ylabel(y_label, labelpad=20, fontsize=12)

#wrap up
plt.legend()
plt.tight_layout()
plt.savefig(name)






