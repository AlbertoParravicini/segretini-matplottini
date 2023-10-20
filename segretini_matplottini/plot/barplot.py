from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator

from segretini_matplottini.utils import (
    add_legend_with_dark_shadow,
    adjust_rows_and_columns_to_number_of_plots,
    create_hex_palette,
    extend_palette,
)
from segretini_matplottini.utils import (
    reset_plot_style as _reset_plot_style,
)
from segretini_matplottini.utils.colors import TWO_TEAL_TONES
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE

DEFAULT_PALETTE = TWO_TEAL_TONES


def _setup_palette(palette: Optional[list[str]], bar_categories: list[str]) -> list[str]:
    if palette is None:
        palette = create_hex_palette(DEFAULT_PALETTE[0], DEFAULT_PALETTE[1], len(bar_categories))
    elif len(palette) > len(bar_categories):
        print(f"⚠️  truncating palette to match the number of categories ({len(palette)} > {len(bar_categories)})")
        palette = palette[: len(bar_categories)]
    elif len(palette) < len(bar_categories):
        print(f"⚠️  extending palette to match the number of categories ({len(palette)} < {len(bar_categories)})")
        palette = extend_palette(palette, len(bar_categories))
    return palette


def _add_legend(ax: Axes, x_to_legend_label_map: dict[str, str], x_axis_categories: list[str]) -> Axes:
    labels = [x_to_legend_label_map.get(m, m) for m in x_axis_categories]
    handles: list[Rectangle] = []
    # Retrieve the colors from the patches of the barplot.
    # If there are more patches than categories,
    # it's because we are dividing the plots by an additional category.
    # In this case, we need to get one every `x_axis_categories` patches.
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    if len(rectangles) == len(x_axis_categories):
        sample_rectangles = rectangles
    elif len(rectangles) > len(x_axis_categories):
        sample_rectangles = rectangles[:: len(rectangles) // len(x_axis_categories)]
    else:
        raise ValueError(
            f"❌ the number of bars ({len(rectangles)}) is less than the number of categories ({len(x_axis_categories)})"
        )
    for r in sample_rectangles:
        assert isinstance(r, Rectangle)
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=r.get_facecolor(),
                edgecolor="#2f2f2f",
                linewidth=0.5,
            )
        )
    add_legend_with_dark_shadow(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=DEFAULT_FONT_SIZE,
        edgecolor="#2f2f2f",
        shadow_offset=1,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=1,
    )
    return ax


def barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylimits: Optional[tuple[float, float]] = None,
    y_axis_ticks_count: int = 6,
    palette: Optional[list[str]] = None,
    add_legend: bool = False,
    x_to_legend_label_map: Optional[dict[str, str]] = None,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.65),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.97,
    bottom_padding: float = 0.34,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a single barplot, given an input dataframe and an `x` and `y` columns.

    :param data: A pandas DataFrame containing the data to plot.
        It must contain a string/categorical column whose name is the value of `x`,
        and a numeric column whose name is the value of `y`.
    :param x: Name of the column in `data` that contains the categories to plot on the x-axis.
    :param y: Name of the column in `data` that contains the values to plot on the y-axis.
    :param xlabel: Label to add to the x-axis, if present.
    :param ylabel: Label to add to the y-axis, if present.
    :param ylimits: Limits of the y-axis. If none, they are inferred by Matplotlib.
    :param y_axis_ticks_count: Number of ticks to show on the y-axis.
    :param palette: A list of colors to use for the bars. If None, use a default palette.
    :param add_legend: If True, add a legend to the plot. Disabled by default,
        since the information is redundant w.r.t. the x-axis tick labels.
    :param x_to_legend_label_map: A dictionary that maps the values of `x` to the labels to show in the legend.
        If a key is missing, keep the original value of `x`.
        Useful when using short-forms for the x-tick labels, due to space constraints.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is approximately half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two barplots per column.
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
    :return: Matplotlib figure and axis containing the plot.
    """

    ##############
    # Setup data #
    ##############

    # Unique categories, each corresponding to a bar;
    bar_categories = list(data[x].unique())

    # Fix missing default values;
    if x_to_legend_label_map is None:
        x_to_legend_label_map = {}

    ##############
    # Setup plot #
    ##############

    # Setup the palette;
    palette = _setup_palette(palette=palette, bar_categories=bar_categories)

    # Initialize figure;
    if reset_plot_style:
        _reset_plot_style(label_pad=2)
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size, dpi=DEFAULT_DPI)
        plt.subplots_adjust(top=top_padding, bottom=bottom_padding, left=left_padding, right=right_padding)
    else:
        _fig = ax.get_figure()
        assert _fig is not None, "❌ the axis has no figure associated"
        fig = _fig
    # Assigning "palette" without assigning "hue" is no longer supported in Seaborn.
    # We get the same effect by setting the "x" variable to "hue", and setting "legend" to False.
    hue = x

    ##################
    # Add main plots #
    ##################

    sns.barplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
        errorbar=None,
        palette=palette,
        saturation=0.9,
        edgecolor="#2f2f2f",
        linewidth=0.5,
        width=0.7,
        legend=False,
    )

    #####################
    # Style fine-tuning #
    #####################

    # Limit y-axis;
    if ylimits is not None:
        ax.set_ylim(ylimits)
    # Add a grid on the y-axis;
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # Increase the number of y-ticks;
    ax.yaxis.set_major_locator(LinearLocator(y_axis_ticks_count))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
    ax.tick_params(axis="y", labelsize=font_size * 0.7)
    # Set axes labels;
    ax.set_xlabel(xlabel if xlabel is not None else "", fontsize=font_size)
    ax.set_ylabel(ylabel if ylabel is not None else "", fontsize=font_size)
    # Set the x-tick labels in every axis;
    ax.tick_params(axis="x", labelsize=font_size * 0.8)

    if add_legend:
        ax = _add_legend(ax, x_to_legend_label_map, bar_categories)

    return fig, ax


def barplots(
    data: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    add_bars_for_averages: bool = False,
    aggregation_function_for_average: Union[str, Callable[[pd.Series], float]] = "mean",
    number_of_rows: Optional[int] = None,
    number_of_columns: Optional[int] = None,
    ylimits: Optional[tuple[float, float]] = None,
    y_axis_ticks_count: int = 6,
    palette: Optional[list[str]] = None,
    x_to_legend_label_map: Optional[dict[str, str]] = None,
    category_to_y_label_map: Optional[dict[str, str]] = None,
    figure_size: tuple[float, float] = (3.5, 2.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.11,
    right_padding: float = 0.98,
    bottom_padding: float = 0.22,
    top_padding: float = 0.95,
    horizontal_spacing: float = 0.65,
    vertical_spacing: float = 0.4,
    reset_plot_style: bool = True,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """
    Plot multiple barplots, given an input dataframe and an `x` column, a `y` column,
    and a `category` column that identifies each barplot.
    The number of columns and rows can be controlled by the caller. If either the number of rows or columns is present,
    the other value is inferred from the number of categories.
    If both values are missing, the number of rows and columns is approximately the square root of the number of plots,
    to provide a grid that is as square as possible.
    If both the number of rows and the number of columns is present, the number of columns is given priority.

    :param data: A pandas DataFrame containing the data to plot.
        It must contain a string/categorical column whose name is the value of `x`,
        a numeric column whose name is the value of `y`,
        and a string/categorical column whose name is the value of `category`.
    :param x: Name of the column in `data` that contains the categories to plot on the x-axis.
    :param y: Name of the column in `data` that contains the values to plot on the y-axis.
    :param category: Name of the column in `data` that contains the categories to plot as different barplots.
    :param add_bars_for_averages: If True, add a group of bars at the start of the plot,
        that represents the average across all categories.
    :param aggregation_function_for_average: Function to use to aggregate the values of `y` across all categories,
        to compute the average. By default, use the mean.
        Either a string representing an aggregation supported by Pandas ("mean", "median", ...) or a Callable
        that can be applied to a Pandas Series.
    :param number_of_rows: Number of rows of the grid of plots. If None, infer it from the number of categories.
    :param number_of_columns: Number of columns of the grid of plots. If None, infer it from the number of categories.
    :param ylimits: Limits of the y-axis. If none, they are inferred by Matplotlib.
    :param y_axis_ticks_count: Number of ticks to show on the y-axis.
    :param palette: A list of colors to use for the bars. If None, use a default palette.
    :param x_to_legend_label_map: A dictionary that maps the values of `x` to the labels to show in the legend.
        If a key is missing, keep the original value of `x`.
        Useful when using short-forms for the x-tick labels, due to space constraints.
    :param category_to_y_label_map: A dictionary that maps the values of `category` to the labels to show
        on the y-axis. If a key is missing, keep the original value of `category`.
    :param figure_size: Size of the figure, in inches. The default is approximately the `\columnwidth` of
        a two-columns template in LaTeX, and the best looking results are achieved with two or three barplots per row,
        and two or three rows.
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
    :return: Matplotlib figure and axis containing the plot.
    """

    ##############
    # Setup data #
    ##############

    _data = data.copy()

    # Plot categories, each corresponding to a barplot;
    categories = list(_data[category].unique())

    # Unique x-axis categories, each corresponding to a bar;
    bar_categories = list(_data[x].unique())

    # Fix missing default values;
    if x_to_legend_label_map is None:
        x_to_legend_label_map = {}
    if category_to_y_label_map is None:
        category_to_y_label_map = {}

    if add_bars_for_averages:
        # Compute the mean;
        averages = _data.groupby(x, sort=False)[y].agg(aggregation_function_for_average).reset_index()
        averages[category] = "Average"
        categories = ["Average"] + categories
        # Create a copy of the input data, prepend the averages at the start;
        _data = pd.concat([averages, _data], ignore_index=True)

    ##############
    # Setup plot #
    ##############

    # Obtain the number of rows and columns to plot;
    _number_of_rows, _number_of_columns = adjust_rows_and_columns_to_number_of_plots(
        number_of_rows=number_of_rows,
        number_of_columns=number_of_columns,
        number_of_plots=len(categories),
    )

    # Setup the palette;
    palette = _setup_palette(palette=palette, bar_categories=bar_categories)

    # Initialize figure;
    if reset_plot_style:
        _reset_plot_style(label_pad=2)
    fig, axes = plt.subplots(_number_of_rows, _number_of_columns, figsize=figure_size, dpi=DEFAULT_DPI, squeeze=False)
    # If number_of_rows == number_of_columns == 1,
    # wrap the axes as if we had multiple axes;
    if isinstance(axes, Axes):
        axes = np.array([axes])
    plt.subplots_adjust(
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        wspace=horizontal_spacing,
        hspace=vertical_spacing,
    )

    ##################
    # Add main plots #
    ##################

    for i, (_category, temp_data) in enumerate(_data.groupby(category, sort=False)):
        ax: plt.Axes = axes.flat[i]
        _, ax = barplot(
            temp_data,
            x=x,
            y=y,
            ylabel=category_to_y_label_map.get(_category, _category),
            ylimits=ylimits,
            add_legend=False,
            y_axis_ticks_count=y_axis_ticks_count,
            ax=ax,
            font_size=font_size,
        )

    #####################
    # Style fine-tuning #
    #####################

    # Delete the extra plots
    for ax in axes.flat[len(categories) :]:
        ax.remove()

    # The legend is added at figure level, but we still to pass one axis to obtain the colors;
    _add_legend(axes.flat[0], x_to_legend_label_map, bar_categories)
    # Convert the axes array to a 2D list.
    # Remove deleted axes, by checking if they no longer have a figure reference
    axes_list: list[list[Axes]] = axes.tolist()
    non_stale_axes_list: list[list[Axes]] = [[ax_j for ax_j in ax_i if ax_j.figure is not None] for ax_i in axes_list]
    return fig, non_stale_axes_list


def barplot_for_multiple_categories(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    add_bars_for_averages: bool = True,
    aggregation_function_for_average: Union[str, Callable[[pd.Series], float]] = "mean",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylimits: Optional[tuple[float, float]] = None,
    y_axis_ticks_count: int = 6,
    palette: Optional[list[str]] = None,
    x_to_legend_label_map: Optional[dict[str, str]] = None,
    hue_category_to_x_tick_label_map: Optional[dict[str, str]] = None,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (3.5, 1.6),  # Roughly half-page \columnwidth in LaTeX
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.13,
    right_padding: float = 0.99,
    bottom_padding: float = 0.36,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a barplot where multiple elements (`hue`) can be compared over
    multiple categories (`x`), for a given metric (`y`).
    For example, one can compare multiple models (as `hue`) over multiple datasets (as `x`),
    for a given metric (as `y`).
    Optionally, add to the left an extra group that represents the average across all categories.
    The function works even if `hue` is missing,
    and for example one wants to simply report the performance of a model over multiple datasets

    :param data: A pandas DataFrame containing the data to plot.
        It must contain a string/categorical column whose name is the value of `x`,
        a numeric column whose name is the value of `y`, and optionally a string/categorical column
        whose name is the value of `hue`.
    :param x: Name of the column in `data` that contains the categories to plot on the x-axis.
    :param y: Name of the column in `data` that contains the values to plot on the y-axis.
    :param hue: Name of the column in `data` that contains the elements to plot as different bars.
    :param add_bars_for_averages: If True, add a group of bars at the start of the plot,
        that represents the average across all categories.
    :param aggregation_function_for_average: Function to use to aggregate the values of `y` across all categories,
        to compute the average. By default, use the mean.
        Either a string representing an aggregation supported by Pandas ("mean", "median", ...) or a Callable
        that can be applied to a Pandas Series.
    :param xlabel: Label to add to the x-axis, if present.
    :param ylabel: Label to add to the y-axis, if present.
    :param ylimits: Limits of the y-axis. If none, they are inferred by Matplotlib.
    :param y_axis_ticks_count: Number of ticks to show on the y-axis.
    :param palette: A list of colors to use for the bars. If None, use a default palette.
    :param x_to_legend_label_map: A dictionary that maps the values of `x` to the labels to show in the legend.
        If a key is missing, keep the original value of `x`.
    :param hue_category_to_x_tick_label_map: A dictionary that maps the values of `hue` to the labels to show
        on the x-axis. If a key is missing, keep the original value of `hue`.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is approximately the `\columnwidth` of
        a two-columns template in LaTeX.
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
    :return: Matplotlib figure and axis containing the plot.
    """

    ##############
    # Setup data #
    ##############

    # Fix missing default values;
    if x_to_legend_label_map is None:
        x_to_legend_label_map = {}
    if hue_category_to_x_tick_label_map is None:
        hue_category_to_x_tick_label_map = {}

    _data = data.copy()
    _data[x] = _data[x].replace(hue_category_to_x_tick_label_map)

    # If no hue is passed, create a fake column to use as hue;
    if hue is None:
        _hue = "hue"
        _data[_hue] = ""
    else:
        _hue = hue

    if add_bars_for_averages:
        # Compute the mean;
        averages = _data.groupby(_hue, sort=False)[y].agg(aggregation_function_for_average).reset_index()
        averages[x] = "Average"
        # Create a copy of the input data, prepend the averages at the start;
        _data = pd.concat([averages, _data], ignore_index=True)

    # List of unique categories, each of which corresponds to a bar category/hue.
    # They are in the order in which they appear in the plot;
    bar_categories = list(_data[_hue].unique())

    ##############
    # Setup plot #
    ##############

    # Setup the palette;
    if palette is None:
        if hue is None:
            palette = [DEFAULT_PALETTE[0]]
        else:
            palette = create_hex_palette(DEFAULT_PALETTE[0], DEFAULT_PALETTE[1], len(bar_categories))
    elif len(palette) > len(bar_categories):
        print(f"⚠️  truncating palette to match the number of categories ({len(palette)} > {len(bar_categories)})")
        palette = palette[: len(bar_categories)]
    elif len(palette) < len(bar_categories):
        print(f"⚠️  extending palette to match the number of categories ({len(palette)} < {len(bar_categories)})")
        palette = extend_palette(palette, len(bar_categories))

    # Initialize figure;
    if reset_plot_style:
        _reset_plot_style(label_pad=3)
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size, dpi=DEFAULT_DPI)
        plt.subplots_adjust(top=top_padding, bottom=bottom_padding, left=left_padding, right=right_padding)
    else:
        _fig = ax.get_figure()
        assert _fig is not None, "❌ the axis has no figure associated"
        fig = _fig

    ##################
    # Add main plots #
    ##################

    # Main barplot;
    sns.barplot(
        data=_data,
        x=x,
        y=y,
        hue=_hue,
        ax=ax,
        errorbar=None,
        width=0.8,
        palette=palette,
        saturation=0.9,
        edgecolor="#2f2f2f",
        linewidth=0.5,
        hue_order=bar_categories,
        legend=False,
    )

    #####################
    # Style fine-tuning #
    #####################

    # Limit y-axis;
    if ylimits is not None:
        ax.set_ylim(ylimits)
    # Add a grid on the y-axis;
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # Increase the number of y-ticks;
    ax.yaxis.set_major_locator(LinearLocator(y_axis_ticks_count))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
    ax.tick_params(axis="y", labelsize=font_size * 0.7)
    # Set axes labels;
    ax.set_xlabel(xlabel if xlabel is not None else "", fontsize=font_size)
    ax.set_ylabel(ylabel if ylabel is not None else "", fontsize=font_size)
    # Set the x-tick labels in every axis;
    ax.tick_params(axis="x", labelsize=font_size * 0.7)
    # Plot a vertical line to separate the average results from the other bar groups;
    if add_bars_for_averages:
        ax.axvline(x=0.5, color="#2f2f2f", linewidth=0.5, linestyle="--")
    # Create a common legend shown below the plot;
    if hue is not None:
        ax = _add_legend(ax, x_to_legend_label_map, data[hue].unique())

    return fig, ax
