# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:43:36 2020

Plot a Roofline model of the performance of one or more algorithms w.r.t. a target architecture;

@author: albyr
"""


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib.lines as lines
import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D

import sys

sys.path.append("..")
from plot_utils import *


SCATTER_SIZE = 14

MARKERS = ["o", "X", "D", "P"]
PALETTE = [COLORS["peach1"], COLORS["g2"], COLORS["bb4"], COLORS["bb5"]]

##############################
##############################


def setup_plot():
    """
    Standard setup of plot style;
    """
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Set style;
    sns.set_style("white", {"ytick.left": True, "xticks.bottom": True, "grid.linewidth": 0.5})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams["axes.titlepad"] = 40
    plt.rcParams["axes.labelpad"] = 2
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.major.pad"] = 5
    plt.rcParams["mathtext.fontset"] = "cm"


def roofline(
    performance: list,
    operational_intensity: list,
    peak_performance: list,
    peak_bandwidth: list,
    ax=None,
    setup: bool = True,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    palette: list = None,
    markers: list = None,
    base_font_size: int = 6,
    xlabel: str = "Operational Intensity [{}/B]",
    ylabel: str = "Performance [{}/s]",
    performance_unit: str = "FLOPS",
    add_bandwidth_label: bool = True,
    bandwidth_label: str = "{:.1f} GB/s",
    add_operational_intensity_label: bool = True,
    operational_intensity_label: str = "OI={:.2f}",
    add_legend: bool = False,
    legend_labels: list = None,
):
    """
    Plot a Roofline model with the specified performance data.
    All input values must be specified with the base measurement unit (e.g. Byte/sec instead of GB/sec)

    Parameters
    ----------
    performance : list of performance points to plot, expressed for example as FLOPS/sec
    operational_intensity : list of operational intensity values
    peak_performance : list of peak performance levels
    peak_bandwidth : list of peak bandwidth values
    ax : existing axis where to plot, useful for example when adding a subplot
    setup : if True, setup the plot with standard font and style
    xmin : minimum value on the x-axis. If not present it is inferred from the data
    xmax : maximum value on the x-axis. If not present it is inferred from the data
    ymin : minimum value on the y-axis. If not present it is inferred from the data
    ymax : maximum value on the y-axis. If not present it is inferred from the data
    palette : one or more colors to use for lines and points
    markers : one or more markers to use for points in the plot
    base_font_size : base font size for labels, used e.g. for legend labels and tick labels
    xlabel : label to place on the x-axis. The default value puts the "performance unit" (e.g. FLOPS) inside it
    ylabel : label to place on the y-axis. The default value puts the "performance unit" (e.g. FLOPS) inside it
    performance_unit : performance measurement unit placed in the labels, e.g. FLOPS or NNZ
    add_bandwidth_label : if True, add bandwidth labels above each peak bandwidth line
    bandwidth_label : label used for bandwidth labels, it should contain a {} marker to dynamically place values
    add_operational_intensity_label : if True, add Operational Intensity labels
    operational_intensity_label : label used for Operational Intensity labels
    add_legend : if True, add a legend
    legend_labels : list of legend labels. If missing, use ??? as placeholder

    Returns
    -------
    ax : axis containing the plot

    """

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

    # Set the plot style;
    if setup:
        setup_plot()

    # Create a plot if no axis is provided;
    if ax is None:
        fig = plt.figure(figsize=(2.2, 1.9))
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(top=0.95, bottom=0.2, left=0.2, right=0.89, hspace=0, wspace=0.5)
        ax = fig.add_subplot(gs[0, 0])

    # Always set log-axes;
    plt.yscale("log")
    plt.xscale("log")

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
        color = palette[i] if palette else PALETTE[0]
        marker = markers[i] if markers else MARKERS[0]

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
            s=SCATTER_SIZE,
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

    # Style settings;

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
    ax.set_yticklabels(labels=[get_exp_label(l) for l in ax.get_yticks()], ha="right", fontsize=base_font_size)
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
                marker=markers[j] if markers else MARKERS[0],
                markeredgewidth=0.5,
                markersize=5,
                label=legend_labels[j],
                markerfacecolor=palette[j] if palette else PALETTE[0],
                markeredgecolor="#2f2f2f",
            )
            for j in range(len(legend_labels))
        ]
        leg = ax.legend(
            custom_lines,
            legend_labels,
            bbox_to_anchor=(0.95, 0),
            fontsize=base_font_size,
            ncol=1,
            loc="lower center",
            handletextpad=0.3,
            columnspacing=0.4,
        )
        leg.set_title(None)
        leg._legend_box.align = "left"
        leg.get_frame().set_facecolor("white")

    return ax


##############################
##############################

if __name__ == "__main__":

    #%% Create a single Roofline model;
    performance = 0.4 * 10**9
    peak_bandwidth = 140.8 * 10**9
    peak_performance = 666 * 10**9
    operational_intensity = 1 / 12

    ax = roofline(
        performance, operational_intensity, peak_performance, peak_bandwidth, add_legend=True, legend_labels="CPU"
    )
    save_plot("../../plots", "roofline.{}")

    #%% Create a fancier Roofline model, with multiple lines and custom settings;
    packet_size = 15
    bandwidth_single_core = 13.2 * 10**9
    bandwidth_total = bandwidth_single_core * 32
    max_cores = 64
    clock = 225 * 10**6
    packet_sizes = [packet_size] * 4
    num_cores = [1, 8, 16, 32]
    num_bits = [20, 20, 20, 20]
    peak_performance = [p * clock * max_cores for p in packet_sizes]
    peak_bandwidth = [bandwidth_single_core * c for c in num_cores]
    operational_intensity = [p / (512 / 8) for p in packet_sizes]
    exec_times_fpga = [108, 12, 5, 3]
    performance = [20 * 10**7 / (p / 1000) for p in exec_times_fpga]

    ax = roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
        palette=PALETTE,
        markers=MARKERS,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in num_cores],
    )
    save_plot("../../plots", "roofline_stacked.{}")

    #%% Create a unique plot composed of 2 separate Rooflines;
    num_col = 2
    num_row = 1
    fig = plt.figure(figsize=(2.2 * num_col, 1.9 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.92, hspace=0, wspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    ax = roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in num_cores],
    )

    performance = [0.4 * 10**9, 27 * 10**9, 20 * 10**7 / (3 / 1000)]
    operational_intensity = [1 / 12, 1 / 12, 0.23]
    peak_performance = [666 * 10**9, 3100 * 10**9, packet_size * clock * max_cores]
    peak_bandwidth = [140.8 * 10**9, 549 * 10**9, 422.4 * 10**9]
    ax = fig.add_subplot(gs[0, 1])
    ax = roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        ylabel="",
        add_legend=True,
        legend_labels=["CPU", "GPU", "FPGA"],
        add_bandwidth_label=False,
        add_operational_intensity_label=False,
    )
    save_plot("../../plots", "roofline_double.{}")
