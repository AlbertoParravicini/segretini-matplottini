#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:40:35 2020

@author: aparravi
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

import sys

sys.path.append("..")
from plot_utils import *


def barplot(res: pd.DataFrame) -> plt.Figure:
    """
    In this example, create a grid barplot that compares an FPGA implementation of the Personalized PageRank algorithm against a CPU implementation.
    The FPGA implementation uses different bit-widths that result in different performance, and performance are evaluated on different dataset.
    For some datasets, it is possible to aggregate results as they have the same size, and display confidence intervals.
    Speedup labels and baseline execution time are also displayed in every plot;

    Parameters
    ----------
    res : a pandas DataFrame

    Returns
    -------
    fig : a matplotlib figure with the plot
    """
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams["axes.titlepad"] = 25
    plt.rcParams["axes.labelpad"] = 9
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.major.pad"] = 1

    # Make sure plot will be displayed with this order;
    sorted_sizes = [100000, 200000, 128000, 81306]
    # Title of each subplot;
    title_labels = [r"$\mathdefault{|E|=10^6}$", r"$\mathdefault{|E|=2 · 10^6}$", "Amazon", "Twitter"]

    num_row = 2
    num_col = len(res.groupby(["V"])) // num_row
    fig = plt.figure(figsize=(2 * num_col, 2.6 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.15, right=0.99, hspace=1, wspace=0.35)

    # Define a color palette, repeat it for each plot;
    palettes = (
        [[COLORS["r1"], COLORS["bb0"], COLORS["bb2"], COLORS["bb3"], COLORS["bb4"], COLORS["bb5"]]] * num_col * num_row
    )

    # Main plot title;
    fig.suptitle("Execution Time Speedup\nw.r.t CPU Baseline", fontsize=16, x=0.05, y=0.99, ha="left")

    # Group plot by dataset size, and sort them in the order specified above;
    groups = res.groupby(["V"])
    groups = sorted(groups, key=lambda x: sorted_sizes.index(x[0]))

    # One plot for each group;
    for i, group in enumerate(groups):
        ax = fig.add_subplot(gs[i // num_row, i % num_row])

        # Replace "float" with "32float" to guarantee the right bar sorting;
        group[1].loc[group[1]["n_bit"] == "float", "n_bit"] = "32float"
        # Sort bars by descending bit-width value;
        data = group[1].sort_values(["n_bit"], ascending=False).reset_index(drop=True)
        ax = sns.barplot(
            x="n_bit",
            y="speedup",
            data=data,
            palette=palettes[i],
            capsize=0.05,
            errwidth=0.8,
            ax=ax,
            edgecolor="#2f2f2f",
        )
        # Remove plot borders at the top and right;
        sns.despine(ax=ax)

        # Fix the name of the x-tick labels;
        ax.set_ylabel("")
        ax.set_xlabel("")
        labels = ax.get_xticklabels()
        cpu_label = int(np.mean(data[data["n_bit"] == "cpu"]["exec_time_ms"]))
        for j, l in enumerate(labels):
            if j == 0:
                l.set_text(f"CPU")
            elif (j == 1) and len(labels) > 5:
                l.set_text("F32")
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", which="major", labelsize=8)

        # Fix the name of the y-tick labels;
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}x"))
        ax.tick_params(axis="y", which="major", labelsize=8)

        # Speedup labels;
        offsets = []
        for b in data["n_bit"].unique():
            offsets += [get_upper_ci_size(data.loc[data["n_bit"] == b, "speedup"], ci=0.80)]
        offsets = [o if not np.isnan(o) else 0.2 for o in offsets]
        # Manually fix some label offsets;
        if i == 0:
            offsets[1] = 0.3
        if i == 1:
            offsets[1] = 0.1
        if i == 3:
            offsets[1] = 0.1
        add_labels(ax, vertical_offsets=offsets, rotation=90, fontsize=10, skip_zero=True)

        # Add graph type;
        ax.annotate(
            f"{title_labels[i]}", xy=(0.6, 1), fontsize=12, ha="center", xycoords="axes fraction", xytext=(0.55, 1.35)
        )
        ax.annotate(
            f"CPU Baseline:", xy=(0.5, 0.0), fontsize=9, ha="center", xycoords="axes fraction", xytext=(0.5, -0.28)
        )
        ax.annotate(
            f"{cpu_label} ms",
            xy=(0.5, 0.0),
            fontsize=9,
            color=COLORS["r4"],
            ha="center",
            xycoords="axes fraction",
            xytext=(0.5, -0.40),
        )

    plt.annotate(
        "Fixed-point Bitwidth", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction"
    )
    plt.annotate(
        "Average Speedup",
        xy=(0.05, 0.5),
        fontsize=14,
        ha="center",
        va="center",
        rotation=90,
        xycoords="figure fraction",
    )
    return fig


if __name__ == "__main__":
    # Run this script from the "src" folder;
    res = pd.read_csv("../../data/barplot_data.csv")
    fig = barplot(res)
    plt.savefig(f"../../plots/barplot.pdf")
