from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from segretini_matplottini.utils import (
    add_legend_with_dark_shadow,
    get_exp_label,
)
from segretini_matplottini.utils import (
    reset_plot_style as _reset_plot_style,
)
from segretini_matplottini.utils.colors import MEGA_PINK, PALETTE_GREEN_TONES_6
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE

MARKERS = ["o", "X", "D", "P"]
# A strong pink color and a few shades of green.
# Skip the first color in the palette to avoid having two colors with the same luminance;
PALETTE = [MEGA_PINK] + PALETTE_GREEN_TONES_6[1::2]


def roofline(
    performance: Union[float, list[float], int, list[int]],
    operational_intensity: Union[float, list[float], int, list[int]],
    peak_performance: Union[float, list[float], int, list[int]],
    peak_bandwidth: Union[float, list[float], int, list[int]],
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    palette: Optional[Union[list[str], str]] = None,
    markers: Optional[Union[list[str], str]] = None,
    scatter_size: float = 14,
    xlabel: str = "Operational intensity [{}/B]",
    ylabel: str = "Performance [{}/s]",
    performance_unit: str = "FLOPS",
    add_bandwidth_label: bool = True,
    bandwidth_label: str = "{:.1f} GB/s",
    add_operational_intensity_label: bool = True,
    operational_intensity_label: str = "OI={:.2f}",
    add_legend: bool = False,
    legend_labels: Optional[Union[str, list[str]]] = None,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (2.8, 2.5),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.2,
    right_padding: float = 0.89,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a Roofline model with the specified performance data.
    All input values must be specified with the base measurement unit (e.g. Byte/sec instead of GB/sec).

    :param performance: List of performance points to plot, expressed for example as FLOPS/sec.
    :param operational_intensity: List of operational intensity values.
    :param peak_performance: List of peak performance levels.
    :param peak_bandwidth: List of peak bandwidth values.
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
    :param ax: Existing axis where to plot, useful for example when adding a subplot.
    :param figure_size: Width and height of the figure, in inches.
    :param font_size: Base font size used in the plot. Font size of titles and tick labels is computed from this value.
    :param left_padding: Padding on the left of the plot, as a fraction of the figure width,
        provided to `plt.subplots_adjust`. A value of 0 means no left padding.
        A value of 0 means no left padding. Applied only if `ax` is None.
    :param right_padding: Padding on the right of the plot, as a fraction of the figure width,
        provided to `plt.subplots_adjust`. Must be >= `left_padding`.
        A value of 1 means no right padding. Applied only if `ax` is None.
    :param bottom_padding: Padding on the bottom of the plot, as a fraction of the figure height,
        provided to `plt.subplots_adjust`. A value of 0 means no bottom padding. Applied only if `ax` is None.
    :param top_padding: Padding on the top of the plot, as a fraction of the figure height,
        provided to `plt.subplots_adjust`. Must be >= `bottom_padding`.
        A value of 1 means no top padding. Applied only if `ax` is None.
    :param reset_plot_style: If True, reset the style of the plot before plotting.
        Disabling it can be useful when plotting on an existing axis rather than creating a new one,
        and the existing axis has a custom style.
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
    if isinstance(performance, (int, float)):
        performance = [performance]
    if isinstance(operational_intensity, (int, float)):
        operational_intensity = [operational_intensity]
    if isinstance(peak_performance, (int, float)):
        peak_performance = [peak_performance]
    if isinstance(peak_bandwidth, (int, float)):
        peak_bandwidth = [peak_bandwidth]
    performance = [float(p) for p in performance]
    operational_intensity = [float(o) for o in operational_intensity]
    peak_performance = [float(p) for p in peak_performance]
    peak_bandwidth = [float(p) for p in peak_bandwidth]
    num_rooflines = len(performance)

    assert num_rooflines == len(performance)
    assert num_rooflines == len(operational_intensity)
    assert num_rooflines == len(peak_performance)
    assert num_rooflines == len(peak_bandwidth)
    # Check that the list of colors and markers is compatible with the number of Rooflines to plot;
    if isinstance(palette, list) and len(palette) < num_rooflines:
        repetitions = (num_rooflines + len(palette) - 1) // len(palette)
        palette *= repetitions
    elif palette and not isinstance(palette, list):
        palette = [palette] * num_rooflines
    if isinstance(markers, list) and len(markers) < num_rooflines:
        repetitions = (num_rooflines + len(markers) - 1) // len(markers)
        markers *= repetitions
    elif markers and not isinstance(markers, list):
        markers = [markers] * num_rooflines

    ##############
    # Setup plot #
    ##############

    # Create a figure for the plot, and adjust margins;
    if reset_plot_style:
        _reset_plot_style(label_pad=2)
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size, dpi=DEFAULT_DPI)
        plt.subplots_adjust(
            top=top_padding,
            bottom=bottom_padding,
            left=left_padding,
            right=right_padding,
        )
    else:
        _fig = ax.get_figure()
        assert _fig is not None, "❌ the axis has no figure associated"
        fig = _fig

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
                fontsize=font_size * 0.75,
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
                label, fontsize=font_size, xy=(operational_intensity[i] * 1.1, ax.get_ylim()[0] * 1.1), ha="left"
            )

    #####################
    # Style fine-tuning #
    #####################

    # Set tick number and parameters on x and y axes;
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False)
    # Set grid on y axis;
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # Make sure that all ticks are visible;
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(True)
        tic.tick2line.set_visible(False)
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(True)
        tic.tick2line.set_visible(False)
    # Set exponential labels on the y axis;
    ax.yaxis.set_major_formatter(lambda x, pos: get_exp_label(x))
    ax.tick_params(labelcolor="#2f2f2f", labelsize=font_size * 0.8, pad=1)
    # Fix ticks on the x axis, ensuring that all minor ticks appear;
    ax.minorticks_on()
    # Set x and y axes labels;
    ax.set_xlabel(xlabel.format(performance_unit) if performance_unit else xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel.format(performance_unit) if performance_unit else ylabel, fontsize=font_size)

    # Add legend;
    if add_legend:
        if not legend_labels:
            legend_labels = ["???"] * num_rooflines
        elif not isinstance(legend_labels, list):
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
            fontsize=font_size,
            ncol=1,
            loc="lower center",
            handletextpad=0.3,
            columnspacing=0.4,
            shadow_offset=1,
            line_width=0.5,
        )

    return fig, ax
