import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse
from mpltools import annotation


def draw_loglog_slope(
    fig, ax, origin, width_inches, slope, color, inverted=False, polygon_kwargs=None, label=True, labelcolor=None, label_kwargs=None, zorder=None
):
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
    log_offset = y1 / (x1**slope)

    print(x1 + width)
    y2 = log_offset * ((x1 + width)**slope)
    height = y2 - y1

    # The vertices of the slope
    a = origin
    b = corner
    c = [x2, y2]

    # Draw the slope triangle
    X = np.array([a, b, c])
    triangle = plt.Polygon(X[:3, :], fill=False, zorder=zorder, **polygon_kwargs)
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
    ax.annotate("$1$", bottom_center, xytext=offset_xlabel, textcoords="offset points", ha="center", va=va_xlabel, zorder=zorder, **label_kwargs)
    ax.annotate(
        f"${slope}$", right_center, xytext=offset_ylabel, textcoords="offset points", ha=ha_ylabel, va="center", zorder=zorder, **label_kwargs
    )



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

