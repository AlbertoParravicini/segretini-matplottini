from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from segretini_matplottini.utils.colors import PALETTE_G, PALETTE_O
from segretini_matplottini.utils.plot_utils import (
    add_legend_with_dark_shadow, extend_palette)
from segretini_matplottini.utils.plot_utils import \
    reset_plot_style as _reset_plot_style


def correlation_scatterplot(
    data: pd.DataFrame,
    x: str = "estimate0",
    y: str = "estimate1",
    hue: Optional[str] = None,
    palette: Optional[list[str]] = None,
    xlabel: str = "Speedup estimate, method A (%)",
    ylabel: str = "Speedup estimate, method B (%)",
    plot_kde: bool = True,
    plot_regression: bool = True,
    highlight_negative_area: bool = True,
    xlimits: Optional[tuple[float, float]] = None,
    ylimits: Optional[tuple[float, float]] = None,
    reset_plot_style: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a detailed correlation analysis between two variables.
    Combine a bivariate density plot, a regression plot and a scatterplot.
    This plot modifies low-level parameters in the regression plot
    which are not directly exposed by the Seaborn API,
    and it adds a rotated label with the same slope as the regression line;

    :param data: DataFrame with 2 numeric columns (`x` and `y`)
    :param x: X-column of the plot.
    :param y: Y-column of the plot.
    :param hue: Column of `data`, if present we split the color of the scatterplot
        according to the unique values in `hue`.
    :param palette: Colors associated to the hue. It should have length equal to the unique values in `hue`.
    :param xlabel: Label added to the x-axis.
    :param ylabel: Label added to the y-axis.
    :param plot_kde: If True, add a seaborn KDE plot with the bivariate density.
    :param plot_regression: If True, add a Seaborn linear regression plot.
    :param xlimits: If specified, truncate the x-axis to this interval.
    :param ylimits: If specified, truncate the y-axis to this interval.
    :param reset_plot_style: If True, reset the style of the plot before plotting.
    :return: Matplotlib figure and axis containing the plot
    """

    ##############
    # Plot setup #
    ##############
    if reset_plot_style:
        _reset_plot_style(label_pad=5)

    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(3.4, 3.1))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.19, right=0.93)
    ax = fig.add_subplot(gs[0, 0])

    # Set axes limits;
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    # Highlight the negative part of the plot and the zero axes;
    if highlight_negative_area:
        ax.axhline(0, color="#2f2f2f", linewidth=0.5, zorder=1)
        ax.axvline(0, color="#2f2f2f", linewidth=0.5, zorder=1)
        # Create a Rectangle patch to highlight the negative part. The rectangle is created as ((start_x, start_y), width, height);
        new_patch = Rectangle(
            (ax.get_xlim()[0], ax.get_ylim()[0]),
            -ax.get_xlim()[0],
            -ax.get_ylim()[0],
            linewidth=0,
            edgecolor=None,
            facecolor="#cccccc",
            alpha=0.4,
            zorder=1,
        )
        ax.add_patch(new_patch)

    #################
    # Main plot #####
    #################

    # Add a density plot for the bivariate distribution;
    if plot_kde:
        ax = sns.kdeplot(
            x=x,
            y=y,
            data=data,
            levels=5,
            color=PALETTE_O[3],
            fill=True,
            alpha=0.5,
            zorder=2,
        )

    # Add a regression plot to highlight the correlation between variables, with 95% confidence intervals;
    if plot_regression:
        ax = sns.regplot(
            x=x,
            y=y,
            data=data,
            color=PALETTE_G[2],
            ax=ax,
            truncate=False,
            scatter=False,
            ci=95,
            line_kws={"linewidth": 0.8, "linestyle": "--", "zorder": 3},
        )
        # Update regression confidence intervals,
        #   to set the confidence bands as semi-transparent and change style and colors of borders;
        plt.setp(ax.collections[-1], facecolor="w", edgecolor=PALETTE_G[2], alpha=0.6, linestyles="--", zorder=3)

    # Add a scatterplot for individual elements of the dataset, and change color based on statistical significance;
    ax = sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        palette=extend_palette(palette, len(data[hue].unique())),
        s=15,
        data=data,
        ax=ax,
        edgecolor="#2f2f2f",
        linewidth=0.5,
        zorder=4,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()  # Hack to remove legend;

    #####################
    # Style fine-tuning #
    #####################

    # Add a label with the R^2 correlation factor. First, obtain coefficients from the linear regression;
    if plot_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[x], data[y])
        angle = np.arctan(slope)
        angle = np.rad2deg(angle)  # Convert slope angle from radians to degrees;
        # Transform angle to adapt to axes with different scale;
        trans_angle = ax.transData.transform_angles([angle], np.array([0, 0]).reshape((1, 2)))[0]
        # Add label with Latex Math font, at the right angle;
        ax.annotate(
            r"$\mathdefault{R^2=" + f"{r_value:.2f}}}$",
            rotation_mode="anchor",
            xy=(0.42, 0.45 * slope + intercept),
            rotation=trans_angle,
            fontsize=8,
            ha="left",
            color="#2f2f2f",
        )

    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.5)
    ax.xaxis.grid(True, linewidth=0.5)

    # Ticks, showing speedup percentage (0% is no speedup);
    ax.yaxis.set_major_locator(plt.LinearLocator(9))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}%")
    ax.tick_params(axis="y", labelsize=6)
    ax.xaxis.set_major_locator(plt.LinearLocator(9))
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}%")
    ax.tick_params(axis="x", labelsize=6)

    # Add legend;
    if hue:
        labels = list(data[hue].unique())
        palette = extend_palette(palette, len(labels))
        custom_lines = [Patch(facecolor=palette[::-1][i], edgecolor="#2f2f2f", label=l) for i, l in enumerate(labels)]
        add_legend_with_dark_shadow(
            ax=ax,
            handles=custom_lines,
            labels=labels,
            bbox_to_anchor=(1, 0.0),
            fontsize=6,
            ncol=len(labels),
            loc="lower right",
            handletextpad=0.3,
            columnspacing=1,
            shadow_offset=1,
        )

    # Add axes labels;
    plt.ylabel(ylabel=ylabel, fontsize=8)
    plt.xlabel(xlabel=xlabel, fontsize=8)

    return fig, ax
