import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

parser = argparse.ArgumentParser("plotter")
parser.add_argument("data", type=str)
parser.add_argument("E", type=float)
parser.add_argument("nu", type=float)
args = parser.parse_args()


def draw_loglog_slope(
    fig,
    ax,
    origin,
    width_inches,
    slope,
    color,
    inverted=False,
    polygon_kwargs=None,
    label=True,
    labelcolor=None,
    label_kwargs=None,
    zorder=None,
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

    y2 = log_offset * ((x1 + width) ** slope)
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
    offset_xlabel = (
        [0.0, -0.33 * label_kwargs["fontsize"]]
        if not inverted
        else [0.0, 0.33 * label_kwargs["fontsize"]]
    )
    offset_ylabel = (
        [0.33 * label_kwargs["fontsize"], 0.0]
        if not inverted
        else [-0.33 * label_kwargs["fontsize"], 0.0]
    )

    # Draw the slope labels
    ax.annotate(
        "$1$",
        bottom_center,
        xytext=offset_xlabel,
        textcoords="offset points",
        ha="center",
        va=va_xlabel,
        zorder=zorder,
        **label_kwargs,
    )
    ax.annotate(
        f"${slope}$",
        right_center,
        xytext=offset_ylabel,
        textcoords="offset points",
        ha=ha_ylabel,
        va="center",
        zorder=zorder,
        **label_kwargs,
    )


# Load CSV data
data = pd.read_csv(args.data)
title = r"$\tilde{E} = $" + str(args.E) + r", $\nu = $" + str(args.nu)

# Filter only converged entries
multigrid_data = data[data["multigrid_converged_no_allocation"] == 1]
standard_data = data[data["standard_converged_no_allocation"] == 1]

# Group by dimension and compute mean and std
multigrid_stats = multigrid_data.groupby("dim")["multigrid_time_no_allocation"].agg(["mean", "std"]).reset_index()
standard_stats = standard_data.groupby("dim")["standard_time_no_allocation"].agg(["mean", "std"]).reset_index()

# Plotting of Runtimes
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    multigrid_stats["dim"],
    multigrid_stats["mean"],
    yerr=multigrid_stats["std"],
    fmt="o-",
    capsize=5,
    label="Multigrid Method",
)
ax.errorbar(
    standard_stats["dim"],
    standard_stats["mean"],
    yerr=standard_stats["std"],
    fmt="s-",
    capsize=5,
    label="Standard Method",
)


# Add labels and legend
draw_loglog_slope(fig, ax, (64*64, 1), 5, 2, "black")
draw_loglog_slope(fig, ax, (256*256, 20), 5, 4, "black")
plt.xlabel("Grid Points")
plt.ylabel("Runtime (seconds)")
plt.xscale("log", base=2)
plt.yscale("log")
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("runtimes.pdf")

# plot only multigrid iterations
multigrid_iterations = (
    multigrid_data.groupby("dim")["multigrid_iterations_no_allocation"].agg(["mean", "std"]).reset_index()
)
fig, ax = plt.subplots()
ax.errorbar(
    multigrid_iterations["dim"],
    multigrid_iterations["mean"],
    yerr=multigrid_iterations["std"],
    fmt="s-",
    capsize=5,
    label="Iterations",
)
plt.legend()
plt.xscale("log", base=2)
plt.yscale("log")
plt.title(title)
plt.savefig("multigrid_iterations.pdf")

# plot only standard iterations
standard_iterations = (
    standard_data.groupby("dim")["standard_iterations_no_allocation"].agg(["mean", "std"]).reset_index()
)
fig, ax = plt.subplots()

ax.errorbar(
    standard_iterations["dim"],
    standard_iterations["mean"],
    yerr=standard_iterations["std"],
    fmt="s-",
    capsize=5,
    label="Iterations",
)
plt.xlabel("Grid Points")
plt.ylabel("Iterations")
plt.legend()
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (64*64, 1000), 5, 2, "black")
plt.title(title)
plt.savefig("standard_iterations.pdf")


#plot WU over dim
standard_wu = (
    standard_data.groupby("dim")["standard_wu"].agg(["mean", "std"]).reset_index()
)
multigrid_wu = (
    standard_data.groupby("dim")["multigrid_wu"].agg(["mean", "std"]).reset_index()
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    multigrid_wu["dim"],
    multigrid_wu["mean"],
    yerr=multigrid_wu["std"],
    fmt="o-",
    capsize=5,
    label="Multigrid Method",
)
ax.errorbar(
    standard_wu["dim"],
    standard_wu["mean"],
    yerr=standard_wu["std"],
    fmt="s-",
    capsize=5,
    label="Standard Method",
)

# Add labels and legend
draw_loglog_slope(fig, ax, (64*64, 1), 5, 2, "black")
draw_loglog_slope(fig, ax, (256*256, 20), 5, 4, "black")
plt.xlabel("Grid Points")
plt.ylabel("WU")
plt.xscale("log", base=2)
plt.yscale("log")
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wu.pdf")