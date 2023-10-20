from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LinearLocator
from scipy import stats

from segretini_matplottini.utils import (
    add_legend_with_dark_shadow,
    extend_palette,
)
from segretini_matplottini.utils import reset_plot_style as _reset_plot_style
from segretini_matplottini.utils.colors import PALETTE_GREEN_TONES_6, TWO_PEACH_TONES
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE


def correlation_scatterplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    plot_kde: bool = True,
    plot_regression: bool = True,
    highlight_negative_area: bool = False,
    scatterplot_palette: Optional[list[str]] = None,
    density_color: str = TWO_PEACH_TONES[1],
    regression_color: str = PALETTE_GREEN_TONES_6[1],
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlimits: Optional[tuple[float, float]] = None,
    ylimits: Optional[tuple[float, float]] = None,
    x_ticks_formatter: Callable[[float, float], str] = lambda x, pos: f"{x * 100:.0f}%",
    y_ticks_formatter: Callable[[float, float], str] = lambda x, pos: f"{x * 100:.0f}%",
    vertical_legend: bool = False,
    legend_position: str = "best",
    label_color: str = "#2f2f2f",
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (3.4, 3.1),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.19,
    right_padding: float = 0.93,
    bottom_padding: float = 0.15,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[plt.Figure, Axes]:
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
    :param plot_kde: If True, add a seaborn KDE plot with the bivariate density.
    :param plot_regression: If True, add a Seaborn linear regression plot.
    :param highlight_negative_area: If True, highlight the negative part of the plot.
    :param scatterplot_palette: Colors associated to the hues of the scatterplot.
        It should have length equal to the unique values in `hue`.
    :param density_color: Color of the bivariate density plot.
    :param regression_color: Color of the lines in the regression plot.
    :param xlabel: Label added to the x-axis.
    :param ylabel: Label added to the y-axis.
    :param xlimits: If specified, truncate the x-axis to this interval.
    :param ylimits: If specified, truncate the y-axis to this interval.
    :param x_ticks_formatter: Callable function used to format x-axis tick labels.
    :param y_ticks_formatter: Callable function used to format y-axis tick labels.
    :param vertical_legend: If True, draw a vertical legend instead of an horizontal one.
    :param legend_position: Position of the legend.
    :param label_color: Color of the linear regression label.
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
    # Setup plot #
    ##############

    # Create a figure for the plot, and adjust margins;
    if reset_plot_style:
        _reset_plot_style(label_pad=5)
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
    assert ax is not None, "❌ the axis is None"

    # Create default color palette;
    if scatterplot_palette is None:
        scatterplot_palette = PALETTE_GREEN_TONES_6[::2]
    if hue is not None:
        scatterplot_palette = extend_palette(scatterplot_palette, len(data[hue].unique()))
    else:
        scatterplot_color = scatterplot_palette = [scatterplot_palette[1]]

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
            color=density_color,
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
            color=regression_color,
            ax=ax,
            truncate=False,
            scatter=False,
            ci=95,
            line_kws={"linewidth": 0.8, "linestyle": "--", "zorder": 4},
        )
        # Update regression confidence intervals,
        #   to set the confidence bands as semi-transparent and change style and colors of borders;
        assert ax is not None, "❌ the axis is None"
        plt.setp(ax.collections[-1], facecolor="w", edgecolor=regression_color, alpha=0.6, linestyles="--", zorder=3)

    # Add a scatterplot for individual elements of the dataset, and change color based on statistical significance;
    ax = sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        palette=scatterplot_palette if hue is not None else None,
        color=scatterplot_color[0] if hue is None else None,
        s=15,
        data=data,
        ax=ax,
        edgecolor="#2f2f2f",
        linewidth=0.5,
        zorder=3,
    )
    assert ax is not None, "❌ the axis is None"
    # Remove the existing legend from the axis, if present
    if ax.get_legend():
        ax.get_legend().remove()

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
        x_coord = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.8 + ax.get_xlim()[0]
        y_coord = x_coord * 1.05 * slope + intercept
        ax.annotate(
            r"$\mathdefault{R^2=" + f"{r_value:.2f}}}$",
            rotation_mode="anchor",
            xy=(x_coord, y_coord),
            rotation=trans_angle,
            fontsize=font_size,
            ha="left",
            color=label_color,
            zorder=100,
        )

    # Turn on the grid;
    ax.grid(axis="both", linestyle="--", linewidth=0.5)

    # Ticks and tick labels;
    ax.xaxis.set_major_locator(LinearLocator(9))
    ax.xaxis.set_major_formatter(x_ticks_formatter)
    ax.yaxis.set_major_locator(LinearLocator(9))
    ax.yaxis.set_major_formatter(y_ticks_formatter)
    ax.tick_params(labelcolor="#2f2f2f", labelsize=font_size * 0.8)
    # Add legend;
    if hue:
        labels = list(data[hue].unique())
        custom_lines = [
            Patch(facecolor=scatterplot_palette[::-1][i], edgecolor="#2f2f2f", label=_l) for i, _l in enumerate(labels)
        ]
        add_legend_with_dark_shadow(
            ax=ax,
            handles=custom_lines,
            labels=labels,
            bbox_to_anchor=(1, 0.0) if legend_position == "lower right" else None,
            fontsize=font_size,
            ncol=1 if vertical_legend else len(labels),
            loc=legend_position,
            handletextpad=0.3,
            columnspacing=1,
            shadow_offset=1,
        )

    # Add axes labels;
    if xlabel is not None:
        plt.xlabel(xlabel=xlabel if xlabel is not None else "", fontsize=font_size)
    if xlabel is not None:
        plt.ylabel(ylabel=ylabel if ylabel is not None else "", fontsize=font_size)
    return fig, ax
