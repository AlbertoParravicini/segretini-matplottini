from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

from segretini_matplottini.utils.plot_utils import (
    add_legend_with_dark_shadow, get_ci_size, reset_plot_style)


def ridgeplot(
    data: pd.DataFrame,
    plot_confidence_intervals: bool = True,
    identifier_column: str = "name",
    column_1: str = "distribution_1",
    column_2: str = "distribution_2",
    xlimits: Optional[tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    number_of_plot_columns: int = 2,
    legend_labels: tuple[str, str] = ("Distribution 1", "Distribution 2"),
    palette: tuple[str, str] = ("#FF6494", "#45E6B0")[::-1],
    compact_layout: bool = True,
) -> sns.FacetGrid:
    """
    Draw a ridgeplot that compares two distributions across different populations.
    For example, the performance of different benchmarks before and after some optimization.
    Or the height of different populations of trees before and after some natural event.

    :param data: The data to plot. A DataFrame that contains numerical columns with names `column_1` and `column_2`.
    :param plot_confidence_intervals: If True, plot 95% confidence intervals centered
        on the mean of each population, along with the distribution.
    :param identifier_column: Name of the column that identifies each sub-population (e.g. "benchmark_name" or "country").
    :param column_1: Name of the column that identifies the first distribution
        to plot in each sub-population (e.g. "tree_height_before_event")
    :param column_2 : Name of the column that identifies the second distribution
        to plot in each sub-population (e.g. "tree_height_after_event")
    :param xlimits: If specified, limit each x-axis to these two values.
    :param number_of_plot_columns: Divide the plot into the specified number of columns.
    :param xlabel: If specified, add this label to the main x-axis.
    :param legend_labels: Labels that identify the two distributions that are being compared.
    :param palette: Two colors used to plot the two distributions.
    :param compact_layout : If True, draw distributions on the same column slightly overlapped, to take less space.
    :return: The Seaborn FacetGrid where the plot is contained.
    """

    ##############
    # Setup data #
    ##############

    # As we are plotting on N columns, we need to explicitely assign the column to each plot;
    _data = data.copy()
    row_identifier = "row_num"
    col_identifier = "col_num"
    plot_identifiers = data[identifier_column].unique()
    b_to_col = {
        b: i // np.ceil(len(plot_identifiers) / number_of_plot_columns) for i, b in enumerate(plot_identifiers)
    }
    b_to_row = {b: i % np.ceil(len(plot_identifiers) / number_of_plot_columns) for i, b in enumerate(plot_identifiers)}
    _data[col_identifier] = _data[identifier_column].replace(b_to_col)
    _data[row_identifier] = _data[identifier_column].replace(b_to_row)

    ##############
    # Setup plot #
    ##############

    # Plotting setup;
    reset_plot_style(title_pad=20, label_pad=10, title_size=22, label_size=14, border_width=1.4)
    # Transparent foreground
    sns.set(rc={"axes.facecolor": (0, 0, 0, 0)})

    # Maximum height of each subplot;
    plot_height = 0.8 if compact_layout else 1.4

    # Initialize the plot.
    # "sharey"is disabled as we don't care that the distributions have the same y-scale.
    # "sharex" is disabled as seaborn would use a single axis object preventing customizations on individual plots.
    # "hue=identifier_column" is necessary to create a mapping over the individual plots,
    # required to set benchmark labels on each axis;
    g = sns.FacetGrid(
        _data,
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
    # Remove grid from every plot, if present
    g.map(lambda label, color: plt.gca().grid(False))

    ##################
    # Add main plots #
    ##################

    # Plot the densities. Plot them twice as the second time we plot just the black contour.
    # "cut" removes values above the threshold; clip=xlimits avoids plotting values outside the margins;
    g.map(
        sns.kdeplot,
        column_1,
        clip_on=False,
        clip=xlimits,
        fill=True,
        alpha=0.8,
        lw=1,
        bw_adjust=1,
        color=palette[0],
        zorder=2,
        cut=10,
    )
    g.map(
        sns.kdeplot,
        column_2,
        clip_on=False,
        clip=xlimits,
        fill=True,
        alpha=0.8,
        lw=1,
        bw_adjust=1,
        color=palette[1],
        zorder=3,
        cut=10,
    )
    g.map(sns.kdeplot, column_1, clip_on=False, clip=xlimits, color="#5f5f5f", lw=1, bw_adjust=1, zorder=2, cut=10)
    g.map(sns.kdeplot, column_2, clip_on=False, clip=xlimits, color="#5f5f5f", lw=1, bw_adjust=1, zorder=3, cut=10)

    if plot_confidence_intervals:
        # Plot a vertical line corresponding to the mean speedup of each benchmark.
        # Pass an unnecessary color argument as name is mapped to hue in the FacetGrid;
        def plot_mean(x, label, color="#2f2f2f"):
            for i, c in enumerate([column_1, column_2]):
                mean_speedup = _data[_data[identifier_column] == label][c].mean()
                plt.axvline(
                    x=mean_speedup,
                    lw=1,
                    clip_on=True,
                    zorder=4,
                    linestyle="dotted",
                    ymax=0.25,
                    color=sns.set_hls_values(palette[i], l=0.3),
                )

        g.map(plot_mean, identifier_column)

        # Plot confidence intervals;
        def plot_ci(x, label, color="#2f2f2f"):
            ax = plt.gca()
            y_max = 0.25 * ax.get_ylim()[1]
            for i, c in enumerate([column_1, column_2]):
                color = sns.set_hls_values(palette[i], l=0.3)
                fillcolor = matplotlib.colors.to_rgba(
                    color, alpha=0.2
                )  # Add alpha to facecolor, while leaving the border opaque;
                upper, lower, _ = get_ci_size(_data[_data[identifier_column] == label][c], get_raw_location=True)
                new_patch = Rectangle(
                    (lower, 0), upper - lower, y_max, linewidth=1, edgecolor=color, facecolor=fillcolor, zorder=4
                )
                ax.add_patch(new_patch)

        g.map(plot_ci, identifier_column)

    #####################
    # Style fine-tuning #
    #####################

    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=1.5, clip_on=False, zorder=4)

    # Fix the horizontal axes so that they are in the specified range (xlimits).
    # Pass an unnecessary label and color arguments as required by FacetGrid.map,
    # as name is mapped to hue;
    if xlimits is not None:

        def set_x_width(label="", color="#2f2f2f"):
            ax = plt.gca()
            ax.set_xlim(left=xlimits[0], right=xlimits[1])

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

    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(
        top=0.98,
        bottom=0.21 if compact_layout else 0.1,
        right=0.98,
        left=0.02,
        hspace=-0.20 if compact_layout else 0.4,
        wspace=0.1,
    )

    # Remove titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    g.set(ylabel=None)

    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(100 * x)}%"

    # Disable y ticks and remove axis;
    g.set(yticks=[])
    g.despine(bottom=True, left=compact_layout)

    # Identify the last axes on each column and update them.
    # We handle the case where the total number of plots is < than rows * columns;
    n_rows = int(_data[row_identifier].max()) + 1
    n_cols = int(_data[col_identifier].max()) + 1
    n_axes = int(n_rows * n_cols)
    n_full_axes = len(_data[identifier_column].unique())
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
    if xlabel:
        for ax in g.axes[-1][:-1]:
            ax.set_xlabel(xlabel, fontsize=18)
        # Set label on the last axis of the last column;
        g.axes[-1 - (n_axes - n_full_axes)][-1].set_xlabel(xlabel, fontsize=18)
    # Hide labels and ticks on empty axes;
    if n_axes > n_full_axes:
        for ax in g.axes[-(n_axes - n_full_axes) :]:
            ax[-1].xaxis.set_ticklabels([])
            for tic in ax[-1].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

    # Add custom legend;
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=l) for i, l in enumerate(legend_labels)]
    leg, _ = add_legend_with_dark_shadow(
        fig=g.fig,
        handles=custom_lines,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=17,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1,
        shadow_offset=3,
        line_width=1,
    )
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor("white")

    return g
