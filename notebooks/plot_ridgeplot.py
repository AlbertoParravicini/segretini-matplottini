#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:14:53 2020

Example of Ridge plot, inspired by https://seaborn.pydata.org/examples/kde_ridgeplot.html

@author: aparravi
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

sys.path.append("..")
from plot_utils import *

##############################
##############################

INPUT_DATA_PATH = "../../data/ridgeplot_data.csv"
PALETTE = ["#F7C0A4", PALETTE_G[2]]

##############################
##############################


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = remove_outliers_df_grouped(data, "exec_time_1_us", ["name"], debug=True)
    data = remove_outliers_df_grouped(data, "exec_time_2_us", ["name"], debug=True)

    # Add relative execution time;
    data["rel_time_1"] = 1
    data["rel_time_2"] = 1
    for name, g in data.groupby("name", as_index=False):
        data.loc[g.index, "rel_time_1"] = g["exec_time_1_us"] / np.mean(g["exec_time_1_us"])
        data.loc[g.index, "rel_time_2"] = g["exec_time_2_us"] / np.mean(g["exec_time_1_us"])

    # As we are plotting on 2 columns, we need to explicitely assign the column to each plot;
    benchmarks = data["name"].unique()
    num_columns = 2
    b_to_col = {b: i // np.ceil(len(benchmarks) / num_columns) for i, b in enumerate(benchmarks)}
    b_to_row = {b: i % np.ceil(len(benchmarks) / num_columns) for i, b in enumerate(benchmarks)}
    data["col_num"] = data["name"].replace(b_to_col)
    data["row_num"] = data["name"].replace(b_to_row)

    return data


def ridgeplot(
    data: pd.DataFrame,
    plot_confidence_intervals: bool = True,
    identifier_column: str = "name",
    column_1: str = "rel_time_1",
    column_2: str = "rel_time_2",
    row_identifier: str = "row_num",
    col_identifier: str = "col_num",
    compact_layout: bool = True,
) -> sns.FacetGrid:
    """
    Draw a ridgeplot that compares two distributions across different populations.
    For example, the performance of different benchmarks before and after some optimization,
    or the height of males and females across different countries;

    Parameters
    ----------
    data : the data to plot
    plot_confidence_intervals : if True, plot 95% confidence intervals centered on the mean of each population, along with the distribution
    identifier_column : name of the column that identifies each sub-population (e.g. "country")
    column_1 : name of the column that identifies the first distribution to plot in each sub-population (e.g. "height_males")
    column_2 : name of the column that identifies the second distribution to plot in each sub-population (e.g. "height_females")
    row_identifier : numeric identifier that assigns each sub-population to a row in the plot
    col_identifier : numeric identifier that assigns each sub-population to a column in the plot
    compact_layout : if True, draw distributions on the same column slightly overlapped, to take less space

    Returns
    -------
    The FacetGrid containing the plot
    """

    # Plotting setup;
    plt.rcdefaults()  # Reset to default settings;
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams["axes.titlepad"] = 20
    plt.rcParams["axes.labelpad"] = 10
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 14

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]

    x_lim = (0.7, 1.3)

    # Maximum height of each subplot;
    plot_height = 0.8 if compact_layout else 1.4

    # Initialize the plot.
    # "sharey"is disabled as we don't care that the distributions have the same y-scale.
    # "sharex" is disabled as seaborn would use a single axis object preventing customizations on individual plots.
    # "hue=identifier_column" is necessary to create a mapping over the individual plots, required to set benchmark labels on each axis;
    g = sns.FacetGrid(
        data,
        hue=identifier_column,
        aspect=8,
        height=plot_height,
        palette=["#2f2f2f"],
        sharex=False,
        sharey=False,
        row=row_identifier,
        col=col_identifier,
    )

    # Plot a vertical line corresponding to speedup = 1;
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=plot_height)

    # Plot the densities. Plot them twice as the second time we plot just the black contour.
    # "cut" removes values above the threshold; clip=x_lim avoids plotting values outside the margins;
    g.map(
        sns.kdeplot,
        column_1,
        clip_on=False,
        clip=x_lim,
        shade=True,
        alpha=0.8,
        lw=1,
        bw_adjust=1,
        color=PALETTE[0],
        zorder=2,
        cut=10,
    )
    g.map(
        sns.kdeplot,
        column_2,
        clip_on=False,
        clip=x_lim,
        shade=True,
        alpha=0.8,
        lw=1,
        bw_adjust=1,
        color=PALETTE[1],
        zorder=3,
        cut=10,
    )
    g.map(sns.kdeplot, column_1, clip_on=False, clip=x_lim, color="#5f5f5f", lw=1, bw_adjust=1, zorder=2, cut=10)
    g.map(sns.kdeplot, column_2, clip_on=False, clip=x_lim, color="#5f5f5f", lw=1, bw_adjust=1, zorder=3, cut=10)

    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=1.2, clip_on=False, zorder=4)

    # Fix the horizontal axes so that they are in the specified range (x_lim);
    def set_x_width(label="", color="#2f2f2f"):
        ax = plt.gca()
        ax.set_xlim(left=x_lim[0], right=x_lim[1])

    g.map(set_x_width)

    # Plot the name of each plot;
    def label(x, label, color="#2f2f2f"):
        ax = plt.gca()
        ax.text(
            0 if compact_layout else 0.01,
            0.15,
            label,
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
        )

    g.map(label, identifier_column)

    if plot_confidence_intervals:
        # Plot a vertical line corresponding to the mean speedup of each benchmark.
        # Pass an unnecessary "color" argument as "name" is mapped to "hue" in the FacetGrid;
        def plot_mean(x, label, color="#2f2f2f"):
            for i, c in enumerate([column_1, column_2]):
                mean_speedup = data[data[identifier_column] == label][c].mean()
                plt.axvline(
                    x=mean_speedup,
                    lw=1,
                    clip_on=True,
                    zorder=4,
                    linestyle="dotted",
                    ymax=0.25,
                    color=sns.set_hls_values(PALETTE[i], l=0.3),
                )

        g.map(plot_mean, identifier_column)

        # Plot confidence intervals;
        def plot_ci(x, label, color="#2f2f2f"):
            ax = plt.gca()
            y_max = 0.25 * ax.get_ylim()[1]
            for i, c in enumerate([column_1, column_2]):
                color = sns.set_hls_values(PALETTE[i], l=0.3)
                fillcolor = matplotlib.colors.to_rgba(
                    color, alpha=0.2
                )  # Add alpha to facecolor, while leaving the border opaque;
                upper, lower, _ = get_ci_size(data[data[identifier_column] == label][c], get_raw_location=True)
                new_patch = Rectangle(
                    (lower, 0), upper - lower, y_max, linewidth=1, edgecolor=color, facecolor=fillcolor, zorder=4
                )
                ax.add_patch(new_patch)

        g.map(plot_ci, identifier_column)

    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(
        top=0.98,
        bottom=0.21 if compact_layout else 0.1,
        right=0.98,
        left=0.02,
        hspace=-0.20 if compact_layout else 0.4,
        wspace=0.1,
    )

    # Titles and labels;
    g.set_titles("")
    g.set(xlabel=None)

    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(100 * x)}%"

    # Disable y ticks and remove axis;
    g.set(yticks=[])
    g.despine(bottom=True, left=compact_layout)

    # Identify the last axes on each column and update them.
    # We handle the case where the total number of plots is < than rows * columns.
    # It is not necessary in this case, but it's useful to have;
    n_rows = int(data[row_identifier].max()) + 1
    n_cols = int(data[col_identifier].max()) + 1
    n_axes = int(n_rows * n_cols)
    n_full_axes = len(data[identifier_column].unique())
    # Set ticks and labels on all axes;
    for i, ax_i in enumerate(g.axes):
        for k, ax_j in enumerate(ax_i):
            if compact_layout and i < len(g.axes) - 1:
                ax_j.xaxis.set_ticklabels([])
            else:
                ax_j.xaxis.set_major_formatter(major_formatter)
                ax_j.tick_params(axis="x", which="major", labelsize=14)
                for tic in ax_j.xaxis.get_major_ticks():
                    tic.tick1line.set_visible(True)
                    tic.tick2line.set_visible(False)
    # Set labels on the last axis of each column (except the last, which could have fewer plots);
    for ax in g.axes[-1][:-1]:
        ax.set_xlabel("Relative Execution Time", fontsize=18)
    # Set label on the last axis of the last column;
    g.axes[-1 - (n_axes - n_full_axes)][-1].set_xlabel("Relative Execution Time", fontsize=18)
    # Hide labels and ticks on empty axes;
    if n_axes > n_full_axes:
        for ax in g.axes[-(n_axes - n_full_axes) :]:
            ax[-1].xaxis.set_ticklabels([])
            for tic in ax[-1].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

    # Add custom legend;
    labels = ["Before transformations", "After transformations"]
    custom_lines = [Patch(facecolor=PALETTE[i], edgecolor="#2f2f2f", label=l) for i, l in enumerate(labels)]
    leg = g.fig.legend(
        custom_lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=17,
        ncol=2,
        handletextpad=0.5,
        columnspacing=0.4,
    )
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor("white")

    return g


##############################
##############################

if __name__ == "__main__":

    # Load data. The input dataframe is a collection of execution times of different benchmarks.
    # Each benchmark has around 100 samples, and it has been measure before transformations ("exec_time_1_us") and after ("exec_time_2_us");
    data = pd.read_csv(INPUT_DATA_PATH)

    # Compute relative execution time before and after transformations and remove outliers.
    # Also assign row/column identifiers to each benchmark for the ridgeplot;
    data = clean_data(data)

    # Plotting;
    g = ridgeplot(data, compact_layout=True)
    # Save the plot;
    save_plot("../../plots", "ridgeplot.{}")

    # Plotting;
    g = ridgeplot(data, compact_layout=False)
    # Save the plot;
    save_plot("../../plots", "ridgeplot_large.{}")
