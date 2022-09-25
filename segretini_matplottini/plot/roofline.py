from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from segretini_matplottini.utils.colors import BB4, BB5, G2, PEACH1
from segretini_matplottini.utils.plot_utils import (
    add_legend_with_dark_shadow, get_exp_label)
from segretini_matplottini.utils.plot_utils import \
    reset_plot_style as _reset_plot_style

MARKERS = ["o", "X", "D", "P"]
PALETTE = [PEACH1, G2, BB4, BB5]


def roofline(
    performance: list[float],
    operational_intensity: list[float],
    peak_performance: list[float],
    peak_bandwidth: list[float],
    ax: Optional[Axis] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    palette: Optional[list[str]] = None,
    markers: Optional[list[str]] = None,
    base_font_size: float = 6,
    scatter_size: float = 14,
    xlabel: str = "Operational Intensity [{}/B]",
    ylabel: str = "Performance [{}/s]",
    performance_unit: str = "FLOPS",
    add_bandwidth_label: bool = True,
    bandwidth_label: str = "{:.1f} GB/s",
    add_operational_intensity_label: bool = True,
    operational_intensity_label: str = "OI={:.2f}",
    add_legend: bool = False,
    legend_labels: list[str] = None,
    reset_plot_style: bool = True,
) -> tuple[Figure, Axis]:
    """
    Plot a Roofline model with the specified performance data.
    All input values must be specified with the base measurement unit (e.g. Byte/sec instead of GB/sec).

    :param performance: List of performance points to plot, expressed for example as FLOPS/sec.
    :param operational_intensity: List of operational intensity values.
    :param peak_performance: List of peak performance levels.
    :param peak_bandwidth: List of peak bandwidth values.
    :param ax: Existing axis where to plot, useful for example when adding a subplot.
    :param xmin: Minimum value on the x-axis. If not present it is inferred from the data.
    :param xmax: Maximum value on the x-axis. If not present it is inferred from the data.
    :param ymin: Minimum value on the y-axis. If not present it is inferred from the data.
    :param ymax: Maximum value on the y-axis. If not present it is inferred from the data.
    :param palette: One or more colors to use for lines and points.
    :param markers: One or more markers to use for points in the plot.
    :param base_font_size: Base font size for labels, used e.g. for legend labels and tick labels.
    :param scatter_size: Size of individual scatter points added to the Roofline.
    :param xlabel: Label to place on the x-axis. The default value puts the "performance unit" (e.g. FLOPS) inside it.
    :param ylabel: Label to place on the y-axis. The default value puts the "performance unit" (e.g. FLOPS) inside it.
    :param performance_unit: Performance measurement unit placed in the labels, e.g. FLOPS.
    :param add_bandwidth_label: If True, add bandwidth labels above each peak bandwidth line.
    :param bandwidth_label: Label used for bandwidth labels, it should contain a {} marker to dynamically place values.
    :param add_operational_intensity_label: If True, add Operational Intensity labels.
    :param operational_intensity_label: Label used for Operational Intensity labels.
    :param add_legend: If True, add a legend.
    :param legend_labels: List of legend labels. If missing, use ??? as placeholder.
    :param reset_plot_style: If True, reset the style of the plot before plotting.
    :return: Matplotlib figure and axis containing the plot
    """

    ##############
    # Setup data #
    ##############

    if palette is None:
        palette = PALETTE
    if markers is None:
        markers = MARKERS

    # Make sure that all inputs are lists;
    if type(performance) is not list:
        performance = [performance]
    if type(operational_intensity) is not list:
        operational_intensity = [operational_intensity]
    if type(peak_performance) is not list:
        peak_performance = [peak_performance]
    if type(peak_bandwidth) is not list:
        peak_bandwidth = [peak_bandwidth]
    num_rooflines = len(performance)

    assert num_rooflines == len(performance)
    assert num_rooflines == len(operational_intensity)
    assert num_rooflines == len(peak_performance)
    assert num_rooflines == len(peak_bandwidth)
    # Check that the list of colors and markers is compatible with the number of Rooflines to plot;
    if type(palette) is list and len(palette) < num_rooflines:
        repetitions = (num_rooflines + len(palette) - 1) / len(palette)
        palette *= repetitions
    elif palette and type(palette) is not list:
        palette = [palette] * num_rooflines
    if type(markers) is list and len(markers) < num_rooflines:
        repetitions = (num_rooflines + len(markers) - 1) / len(markers)
        markers *= repetitions
    elif markers and type(markers) is not list:
        markers = [markers] * num_rooflines

    ##############
    # Setup plot #
    ##############

    if reset_plot_style:
        _reset_plot_style(
            grid_linewidth=0.5, title_pad=40, label_pad=2, title_size=22, label_size=14, xtick_major_pad=5
        )

    # Create a plot if no axis is provided;
    if ax is None:
        fig = plt.figure(figsize=(2.2, 1.9))
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(top=0.95, bottom=0.2, left=0.2, right=0.89, hspace=0, wspace=0.5)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig = plt.gcf()

    # Always set log-axes;
    plt.yscale("log")
    plt.xscale("log")

    #################
    # Add rooflines #
    #################

    for i in range(num_rooflines):
        # Compute points required for plotting the Roofline;
        line_cross = peak_performance[i] / peak_bandwidth[i]
        line_cross_op_intensity = peak_bandwidth[i] * operational_intensity[i]

        # Define limits for plotting;
        _xmin = operational_intensity[i] / 5 if not xmin else xmin
        _xmax = (
            np.max([ax.get_xlim()[1], line_cross * 10]) if not xmax else xmax
        )  # Show 10 times more than the minimum operational intensity that causes a compute-bound algorithm;
        _ymin = (
            peak_performance[i] / 10000 if not ymin else ymin
        )  # By default, show up to 1/10000 of peak performance;
        _ymax = np.max([ax.get_ylim()[1], peak_performance[i] * 1.4]) if not ymax else ymax

        # Pick Roofline plot color;
        color = palette[i]
        marker = markers[i]

        # Plot lines;
        plt.plot(
            [operational_intensity[i], operational_intensity[i]],
            [0, line_cross_op_intensity],
            color="#757575",
            linewidth=0.5,
            linestyle="--",
            zorder=0,
        )
        plt.plot(
            [0, _xmax],
            [peak_performance[i], peak_performance[i]],
            color="#757575",
            linewidth=0.5,
            linestyle="--",
            zorder=0,
        )
        plt.plot([line_cross, _xmax], [peak_performance[i], peak_performance[i]], color=color, linewidth=1, zorder=1)
        plt.plot([0, line_cross], [0, peak_performance[i]], color=color, linewidth=1, zorder=1)
        plt.scatter(
            operational_intensity[i],
            performance[i],
            color=color,
            edgecolors="#2f2f2f",
            marker=marker,
            s=scatter_size,
            zorder=2,
            linewidth=0.3,
        )

        # Enforce axes limits;
        ax.set_xlim((_xmin, _xmax))
        ax.set_ylim((_ymin, _ymax))

        # Add bandwidth labels;
        if add_bandwidth_label:
            x_loc = 1.5 * _xmin
            label = bandwidth_label.format(peak_bandwidth[i] / 10**9)
            tan = peak_performance[i] / line_cross
            angle = np.arctan(tan)
            angle = np.rad2deg(angle)
            trans_angle = ax.transData.transform_angles([angle], np.array([0, 0]).reshape((1, 2)))[0]
            ax.annotate(
                label,
                fontsize=base_font_size * 0.75,
                xy=(x_loc, peak_bandwidth[i] * x_loc * 1.1),
                ha="left",
                rotation_mode="anchor",
                rotation=trans_angle,
                color="#2f2f2f",
            )
        # Add operational intensity labels;
        if add_operational_intensity_label:
            label = operational_intensity_label.format(operational_intensity[i])
            ax.annotate(
                label, fontsize=base_font_size, xy=(operational_intensity[i] * 1.1, ax.get_ylim()[0] * 2), ha="left"
            )

    #####################
    # Style fine-tuning #
    #####################

    # Set tick number and parameters on x and y axes;
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False)
    # Set grid on y axis;
    ax.xaxis.grid(False)
    ax.yaxis.grid(linewidth=0.5)
    # Make sure that all ticks are visible;
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(True)
        tic.tick2line.set_visible(False)
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(True)
        tic.tick2line.set_visible(False)
    # Set exponential labels on the y axis;
    ax.yaxis.set_major_formatter(lambda x, pos: get_exp_label(x))
    ax.tick_params(axis="y", labelsize=base_font_size)
    # Fix ticks on the x axis, ensuring that all minor ticks appear;
    ax.tick_params(labelcolor="black", labelsize=base_font_size, pad=1)
    ax.minorticks_on()
    # Set x and y axes labels;
    ax.set_xlabel(xlabel.format(performance_unit) if performance_unit else xlabel, fontsize=base_font_size + 1)
    ax.set_ylabel(ylabel.format(performance_unit) if performance_unit else ylabel, fontsize=base_font_size + 1)

    # Add legend;
    if add_legend:
        if not legend_labels:
            legend_labels = ["???"] * num_rooflines
        elif type(legend_labels) is not list:
            legend_labels = [legend_labels]
        if len(legend_labels) < num_rooflines:
            legend_labels += ["???"] * (num_rooflines - len(legend_labels))
        custom_lines = [
            Line2D(
                [],
                [],
                color="white",
                marker=markers[j],
                markeredgewidth=0.5,
                markersize=5,
                label=legend_labels[j],
                markerfacecolor=palette[j],
                markeredgecolor="#2f2f2f",
            )
            for j in range(len(legend_labels))
        ]
        _, ax = add_legend_with_dark_shadow(
            ax=ax,
            labels=legend_labels,
            handles=custom_lines,
            bbox_to_anchor=(0.9, 0),
            fontsize=base_font_size,
            ncol=1,
            loc="lower center",
            handletextpad=0.3,
            columnspacing=0.4,
            shadow_offset=1,
            line_width=0.5,
        )

    return fig, ax
