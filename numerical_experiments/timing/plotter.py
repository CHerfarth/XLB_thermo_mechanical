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
vcycle_data = data[data["vcycle_converged_no_allocation"] == 1]
wcycle_data = data[data["wcycle_converged_no_allocation"] == 1]
standard_data = data[data["standard_converged_no_allocation"] == 1]

# Group by dimension and compute mean and std
vcycle_stats = (
    vcycle_data.groupby("dim")["vcycle_time_no_allocation"].agg(["mean", "std"]).reset_index()
)
wcycle_stats = (
    wcycle_data.groupby("dim")["wcycle_time_no_allocation"].agg(["mean", "std"]).reset_index()
)
standard_stats = (
    standard_data.groupby("dim")["standard_time_no_allocation"].agg(["mean", "std"]).reset_index()
)

# Plotting of Runtimes
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    vcycle_stats["dim"],
    vcycle_stats["mean"],
    yerr=vcycle_stats["std"],
    fmt="o-",
    capsize=5,
    label="V-Cycle",
    color="blue",
)
ax.errorbar(
    wcycle_stats["dim"],
    wcycle_stats["mean"],
    yerr=wcycle_stats["std"],
    fmt="d-",
    capsize=5,
    label="W-Cycle",
    color="green",
)
ax.errorbar(
    standard_stats["dim"],
    standard_stats["mean"],
    yerr=standard_stats["std"],
    fmt="s-",
    capsize=5,
    label="Standard Method",
    color="red"
)


# Add labels and legend
plt.xlabel("n",fontsize=20)
plt.ylabel("Runtime [seconds]",fontsize=20)
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (32, 0.1), 1, 2, "black")
draw_loglog_slope(fig, ax, (170, 2), 1, 4, "black")
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("runtimes.pdf")

# only plot standard runtimes
fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(
    standard_stats["dim"],
    standard_stats["mean"],
    yerr=standard_stats["std"],
    fmt="s-",
    capsize=5,
    label="Standard Method",
    color="red",
)
# Add labels and legend
plt.xlabel("n",fontsize=20)
plt.ylabel("Runtime (seconds)",fontsize=20)
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (32, 0.1), 1, 2, "black")
draw_loglog_slope(fig, ax, (170, 2), 1, 4, "black")
plt.title(title)
plt.grid(True)
plt.tight_layout()
plt.savefig("runtimes_only_standard.pdf")

# plot only multigrid iterations
vcycle_iterations = (
    vcycle_data.groupby("dim")["vcycle_iterations_no_allocation"]
    .agg(["mean", "std"])
    .reset_index()
)
wcycle_iterations = (
    wcycle_data.groupby("dim")["wcycle_iterations_no_allocation"]
    .agg(["mean", "std"])
    .reset_index()
)
fig, ax = plt.subplots()
ax.errorbar(
    vcycle_iterations["dim"],
    vcycle_iterations["mean"],
    yerr=vcycle_iterations["std"],
    fmt="o-",
    capsize=5,
    label="V-Cycle",
    color="blue",
)
ax.errorbar(
    wcycle_iterations["dim"],
    wcycle_iterations["mean"],
    yerr=wcycle_iterations["std"],
    fmt="d-",
    capsize=5,
    label="W-Cycle",
    color="green",
)
plt.xscale("log", base=2)
plt.ylim(bottom=0, top=50)
plt.xlabel("n",fontsize=20)
plt.ylabel("Iterations",fontsize=20)
plt.title(title)
plt.legend()
plt.savefig("multigrid_iterations.pdf")

# plot only standard iterations
standard_iterations = (
    standard_data.groupby("dim")["standard_iterations_no_allocation"]
    .agg(["mean", "std"])
    .reset_index()
)
fig, ax = plt.subplots()

ax.errorbar(
    standard_iterations["dim"],
    standard_iterations["mean"],
    yerr=standard_iterations["std"],
    fmt="s-",
    capsize=5,
    label="Iterations",
    color="red",
)
plt.xlabel("n",fontsize=20)
plt.ylabel("Iterations",fontsize=20)
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (64, 1000), 1, 2, "black")
plt.title(title)
plt.savefig("standard_iterations.pdf")


# plot WU over dim
standard_wu = standard_data.groupby("dim")["standard_wu"].agg(["mean", "std"]).reset_index()
vcycle_wu = vcycle_data.groupby("dim")["vcycle_wu"].agg(["mean", "std"]).reset_index()
wcycle_wu = wcycle_data.groupby("dim")["wcycle_wu"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    vcycle_wu["dim"],
    vcycle_wu["mean"],
    yerr=vcycle_wu["std"],
    fmt="o-",
    capsize=5,
    label="V-Cycle",
    color="blue",
)
ax.errorbar(
    wcycle_wu["dim"],
    wcycle_wu["mean"],
    yerr=wcycle_wu["std"],
    fmt="d-",
    capsize=5,
    label="W-Cycle",
    color="green",
)
ax.errorbar(
    standard_wu["dim"],
    standard_wu["mean"],
    yerr=standard_wu["std"],
    fmt="s-",
    capsize=5,
    label="Standard Method",
    color="red",
)

# Add labels and legend
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (64 * 64, 1), 1, 2, "black")
plt.xlabel("n",fontsize=20)
plt.ylabel("WU",fontsize=20)
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wu.pdf")

# plot times (with allocation)
vcycle_time_no_allocation = vcycle_data.groupby("dim")["vcycle_time_no_allocation"].agg(["mean", "std"]).reset_index()
vcycle_time_with_allocation = vcycle_data.groupby("dim")["vcycle_time_with_allocation"].agg(["mean", "std"]).reset_index()
wcycle_time_no_allocation = wcycle_data.groupby("dim")["wcycle_time_no_allocation"].agg(["mean", "std"]).reset_index()
wcycle_time_with_allocation = wcycle_data.groupby("dim")["wcycle_time_with_allocation"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    vcycle_time_no_allocation["dim"],
    vcycle_time_no_allocation["mean"],
    yerr=vcycle_time_no_allocation["std"],
    fmt="o-",
    capsize=5,
    label="V-Cycle (allocation time not included)",
    color="blue",
)
ax.errorbar(
    vcycle_time_with_allocation["dim"],
    vcycle_time_with_allocation["mean"],
    yerr=vcycle_time_with_allocation["std"],
    fmt="o--",
    capsize=5,
    label="V-Cycle (with allocation time)",
    color="blue",
)
ax.errorbar(
    wcycle_time_no_allocation["dim"],
    wcycle_time_no_allocation["mean"],
    yerr=wcycle_time_no_allocation["std"],
    fmt="d-",
    capsize=5,
    label="W-Cycle (allocation time not included)",
    color="green",
)
ax.errorbar(
    wcycle_time_with_allocation["dim"],
    wcycle_time_with_allocation["mean"],
    yerr=wcycle_time_with_allocation["std"],
    fmt="d--",
    capsize=5,
    label="W-Cycle (with allocation time)",
    color="green",
)

# Add labels and legend
plt.xscale("log", base=2)
plt.yscale("log")
draw_loglog_slope(fig, ax, (64 * 64, 1), 1, 2, "black")
plt.xlabel("n",fontsize=20)
plt.ylabel("Runtime [seconds]",fontsize=20)
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("allocation_v_no_allocation.pdf")
