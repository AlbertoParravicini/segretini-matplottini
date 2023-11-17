import warnings
from typing import Any, Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle

from segretini_matplottini.utils import (
    add_legend_with_dark_shadow,
    get_ci_size,
)
from segretini_matplottini.utils import (
    reset_plot_style as _reset_plot_style,
)
from segretini_matplottini.utils.colors import GREEN_AND_PINK_TONES
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE


def _grid_map(g: sns.FacetGrid, func: Callable, *args: Any, **kwargs: Any) -> sns.FacetGrid:
    """
    Wrap FacetGrid.map to suppress UserWarnings.
    Since FacetGrid forces a tight layout, and we get warnings when axes are partially overlapped,
    we suppress the warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return g.map(func, *args, **kwargs)


def _setup_data(
    data: pd.DataFrame,
    identifier_column: str,
    number_of_plot_columns: int,
    row_identifier: str,
    col_identifier: str,
) -> pd.DataFrame:
    # As we are plotting on N columns, we need to explicitely assign the column to each plot;
    _data = data.copy()
    plot_identifiers = data[identifier_column].unique()
    b_to_col = {
        b: i // np.ceil(len(plot_identifiers) / number_of_plot_columns) for i, b in enumerate(plot_identifiers)
    }
    b_to_row = {b: i % np.ceil(len(plot_identifiers) / number_of_plot_columns) for i, b in enumerate(plot_identifiers)}
    _data[col_identifier] = _data[identifier_column].replace(b_to_col)
    _data[row_identifier] = _data[identifier_column].replace(b_to_row)
    return _data


def _setup_plot(
    data: pd.DataFrame,
    plot_height: float,
    aspect_ratio: float,
    vertical_line_y_max: float,
    identifier_column: str,
    row_identifier: str,
    col_identifier: str,
    reset_plot_style: bool,
) -> sns.FacetGrid:
    # Plotting setup;
    if reset_plot_style:
        _reset_plot_style(label_pad=5, border_width=0.8)
    # Transparent foreground
    sns.set(rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the plot.
    # "sharey"is disabled as we don't care that the distributions have the same y-scale.
    # "sharex" is disabled as seaborn would use a single axis object preventing customizations on individual plots.
    # "hue=identifier_column" is necessary to create a mapping over the individual plots,
    # required to set labels on each axis;
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        g = sns.FacetGrid(
            data,
            hue=identifier_column,
            aspect=aspect_ratio,
            height=plot_height,
            palette=["#2f2f2f"],
            sharex=False,
            sharey=False,
            row=row_identifier,
            col=col_identifier,
        )

    # Plot a vertical line corresponding to value = 1;
    _grid_map(g, plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=vertical_line_y_max)
    # Remove grid from every plot, if present
    _grid_map(g, lambda label, color: plt.gca().grid(False))
    return g


def _plot_densities(
    g: sns.FacetGrid,
    column_1: str,
    column_2: str,
    xlimits: Optional[tuple[float, float]],
    palette: tuple[str, str],
    line_width: float,
) -> None:
    _grid_map(
        g,
        sns.kdeplot,
        column_1,
        clip_on=False,
        clip=xlimits,
        fill=True,
        alpha=0.8,
        lw=line_width,
        bw_adjust=1,
        color=palette[0],
        zorder=2,
        cut=10,
    )
    _grid_map(
        g,
        sns.kdeplot,
        column_2,
        clip_on=False,
        clip=xlimits,
        fill=True,
        alpha=0.8,
        lw=line_width,
        bw_adjust=1,
        color=palette[1],
        zorder=3,
        cut=10,
    )
    _grid_map(
        g,
        sns.kdeplot,
        column_1,
        clip_on=False,
        clip=xlimits,
        color="#5f5f5f",
        lw=line_width,
        bw_adjust=1,
        zorder=2,
        cut=10,
    )
    _grid_map(
        g,
        sns.kdeplot,
        column_2,
        clip_on=False,
        clip=xlimits,
        color="#5f5f5f",
        lw=line_width,
        bw_adjust=1,
        zorder=3,
        cut=10,
    )


def _style_fine_tuning(
    g: sns.FacetGrid,
    xlimits: Optional[tuple[float, float]],
    x_axis_line_width: float,
) -> None:
    # Plot the horizontal line below the densities;
    _grid_map(g, plt.axhline, y=0, lw=x_axis_line_width, clip_on=False, zorder=4)

    # Fix the horizontal axes so that they are in the specified range (xlimits).
    # Pass an unnecessary label and color arguments as required by FacetGrid.map,
    # as name is mapped to hue;
    if xlimits is not None:
        x_left, x_right = xlimits

        def set_x_width(label: str = "", color: str = "#2f2f2f") -> None:
            ax: Axes = plt.gca()
            ax.set_xlim(left=x_left, right=x_right)

        _grid_map(g, set_x_width)

    # Remove titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    g.set(ylabel=None)


def _add_individual_plot_labels(
    g: sns.FacetGrid,
    plot_label_x_position: float,
    plot_label_y_position: float,
    font_size: float,
    identifier_column: str,
) -> None:
    # Plot the name of each plot;
    def label(x: float, label: str, color: str = "#2f2f2f") -> None:
        ax: Axes = plt.gca()
        ax.text(
            plot_label_x_position,
            plot_label_y_position,
            label,
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size,
        )

    _grid_map(g, label, identifier_column)


@ticker.FuncFormatter
def _x_ticks_major_formatter(x: float, pos: str) -> str:
    """
    Write the x-axis tick labels using percentages.
    """
    return f"{int(100 * x)}%"


def _add_x_labels(axes: list[list[Axes]], font_size: float, xlabel: str, n_axes: int, n_full_axes: int) -> None:
    # Set labels on the last axis of each column (except the last, which could have fewer plots);
    for ax in axes[-1][:-1]:
        assert isinstance(ax, Axes)
        ax.set_xlabel(xlabel, fontsize=font_size)
    # Set label on the last axis of the last column;
    axes[-1 - (n_axes - n_full_axes)][-1].set_xlabel(xlabel, fontsize=font_size)


def _remove_labels_in_empty_axes(axes: list[list[Axes]], n_axes: int, n_full_axes: int) -> None:
    for ax_i in axes[-(n_axes - n_full_axes) :]:
        ax_j: Axes = ax_i[-1]
        assert isinstance(ax_j, Axes)
        ax_j.xaxis.set_ticklabels([])
        for tic in ax_j.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)


def _set_x_labels(
    xlabel: Optional[str],
    data: pd.DataFrame,
    axes: list[list[Axes]],
    identifier_column: str,
    row_identifier: str,
    col_identifier: str,
    font_size: float,
) -> None:
    # Set labels on the last axis of each column (except the last, which could have fewer plots);
    n_rows = int(data[row_identifier].max()) + 1
    n_cols = int(data[col_identifier].max()) + 1
    n_axes = int(n_rows * n_cols)
    n_full_axes = len(data[identifier_column].unique())
    if xlabel:
        _add_x_labels(axes=axes, font_size=font_size, xlabel=xlabel, n_axes=n_axes, n_full_axes=n_full_axes)
    # Hide labels and ticks on empty axes;
    if n_axes > n_full_axes:
        _remove_labels_in_empty_axes(axes=axes, n_axes=n_axes, n_full_axes=n_full_axes)


def _add_legend(palette: tuple[str, str], legend_labels: tuple[str, str], g: sns.FacetGrid, font_size: int) -> None:
    custom_lines = [Patch(facecolor=palette[i], edgecolor="#2f2f2f", label=_l) for i, _l in enumerate(legend_labels)]
    leg, _ = add_legend_with_dark_shadow(
        fig=g.fig,
        handles=custom_lines,
        labels=list(legend_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=font_size,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1,
        shadow_offset=1,
    )
    leg.set_title("")
    leg.set_alignment("left")
    leg.get_frame().set_facecolor("white")


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
    palette: tuple[str, str] = GREEN_AND_PINK_TONES,
    plot_height: float = 0.7,
    aspect_ratio: float = 8,
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.02,
    right_padding: float = 0.98,
    bottom_padding: float = 0.1,
    top_padding: float = 0.98,
    horizontal_spacing: float = 0.1,
    vertical_spacing: float = 0.4,
    reset_plot_style: bool = True,
) -> sns.FacetGrid:
    """
    Draw a ridgeplot that compares two distributions across different populations.
    For example, the performance of different benchmarks before and after some optimization.
    Or the height of different populations of trees before and after some natural event.
    Inspired by https://seaborn.pydata.org/examples/kde_ridgeplot.html

    The total figure size is computed as follows:
    * `number_of_plots = len(data[identifier_column]).unique()`
    * `number_of_rows = ceil(number_of_plots / number_of_plot_columns)`
    * `width = number_of_plot_columns * plot_height * aspect_ratio`
    * `height = number_of_rows * plot_height`
    For example, to obtain a square figure of shape `(3.5, 3.5)`:
    * `plot_height = 3.5 / number_of_plots`
    * `aspect_ratio = number_of_plots`

    :param data: The data to plot. A DataFrame that contains numerical columns with names `column_1` and `column_2`.
    :param plot_confidence_intervals: If True, also plot the 95% confidence intervals
        of the sample mean of each population.
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
    :param plot_height: Height of each plot.
    :param aspect_ratio: Aspect ratio of each plot.
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
    :param horizontal_spacing: Spacing between columns of the plot, as a fraction of the figure width,
        provided to `plt.subplots_adjust` as `wspace`. A value of 0 means no spacing. Applied only if `ax` is None.
    :param vertical_spacing: Spacing between rows of the plot, as a fraction of the figure height,
        provided to `plt.subplots_adjust` as `hspace`. A value of 0 means no spacing. Applied only if `ax` is None.
    :param reset_plot_style: If True, reset the style of the plot before plotting.
        Disabling it can be useful when plotting on an existing axis rather than creating a new one,
        and the existing axis has a custom style.
    :return: The Seaborn FacetGrid where the plot is contained.
    """

    ##############
    # Setup data #
    ##############

    # As we are plotting on N columns, we need to explicitely assign the column to each plot;
    row_identifier = "row_num"
    col_identifier = "col_num"
    _data = _setup_data(
        data=data,
        identifier_column=identifier_column,
        number_of_plot_columns=number_of_plot_columns,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
    )

    ##############
    # Setup plot #
    ##############

    g = _setup_plot(
        data=_data,
        identifier_column=identifier_column,
        plot_height=plot_height,
        aspect_ratio=aspect_ratio,
        vertical_line_y_max=0.9,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
        reset_plot_style=reset_plot_style,
    )
    axes: list[list[Axes]] = g.axes.tolist()

    ##################
    # Add main plots #
    ##################

    # Plot the densities. Plot them twice as the second time we plot just the black contour.
    # "cut" removes values above the threshold; clip=xlimits avoids plotting values outside the margins;
    _plot_densities(g=g, column_1=column_1, column_2=column_2, xlimits=xlimits, palette=palette, line_width=0.7)

    if plot_confidence_intervals:
        # Plot a vertical line corresponding to the mean speedup of each benchmark.
        # Pass an unnecessary color argument as name is mapped to hue in the FacetGrid;
        def plot_mean(x: float, label: str, color: str = "#2f2f2f") -> None:
            for i, c in enumerate([column_1, column_2]):
                mean_speedup = _data[_data[identifier_column] == label][c].mean()
                plt.axvline(
                    x=mean_speedup,
                    lw=0.8,
                    clip_on=True,
                    zorder=4,
                    linestyle="dotted",
                    ymax=0.25,
                    color=sns.set_hls_values(palette[i], l=0.3),
                )

        _grid_map(g, plot_mean, identifier_column)

        # Plot confidence intervals;
        def plot_ci(x: float, label: str, color: str = "#2f2f2f") -> None:
            ax: Axes = plt.gca()
            y_max = 0.25 * ax.get_ylim()[1]
            for i, c in enumerate([column_1, column_2]):
                color = sns.set_hls_values(palette[i], l=0.3)
                fillcolor = matplotlib.colors.to_rgba(
                    color, alpha=0.2
                )  # Add alpha to facecolor, while leaving the border opaque;
                upper, lower, _ = get_ci_size(_data[_data[identifier_column] == label][c], get_raw_location=True)
                new_patch = Rectangle(
                    (lower, 0), upper - lower, y_max, linewidth=0.8, edgecolor=color, facecolor=fillcolor, zorder=4
                )
                ax.add_patch(new_patch)

        _grid_map(g, plot_ci, identifier_column)

    #####################
    # Style fine-tuning #
    #####################

    _style_fine_tuning(
        g=g,
        xlimits=xlimits,
        x_axis_line_width=1,
    )
    _add_individual_plot_labels(
        g=g,
        plot_label_x_position=0.003,
        plot_label_y_position=0.2,
        identifier_column=identifier_column,
        font_size=font_size * 1.2,
    )

    # Disable y ticks and remove axis;
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Set the ticks and tick labels;
    for ax_i in axes:
        for ax_j in ax_i:
            assert isinstance(ax_j, Axes)
            ax_j.xaxis.set_major_formatter(_x_ticks_major_formatter)
            ax_j.tick_params(axis="x", which="major", labelsize=font_size * 0.8, width=1, length=4, color="#2f2f2f")
            for tic in ax_j.xaxis.get_major_ticks():
                tic.tick1line.set_visible(True)
                tic.tick2line.set_visible(False)
    # Set labels on the last axis of each column (except the last, which could have fewer plots).
    # Hide labels and ticks on empty axes;
    _set_x_labels(
        xlabel=xlabel,
        data=_data,
        axes=axes,
        identifier_column=identifier_column,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
        font_size=font_size,
    )

    # Fix the borders. This must be done here as the previous operations update the default values;
    plt.subplots_adjust(
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        hspace=vertical_spacing,
        wspace=horizontal_spacing,
    )

    # Add custom legend;
    _add_legend(palette=palette, legend_labels=legend_labels, g=g, font_size=font_size)

    return g


def ridgeplot_compact(
    data: pd.DataFrame,
    identifier_column: str = "name",
    column_1: str = "distribution_1",
    column_2: str = "distribution_2",
    xlimits: Optional[tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    number_of_plot_columns: int = 2,
    legend_labels: tuple[str, str] = ("Distribution 1", "Distribution 2"),
    palette: tuple[str, str] = GREEN_AND_PINK_TONES,
    plot_height: float = 0.5,
    aspect_ratio: float = 8,
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.02,
    right_padding: float = 0.98,
    bottom_padding: float = 0.17,
    top_padding: float = 0.98,
    horizontal_spacing: float = 0.1,
    vertical_spacing: float = -0.2,
    reset_plot_style: bool = True,
) -> sns.FacetGrid:
    """
    Draw a ridgeplot that compares two distributions across different populations.
    For example, the performance of different benchmarks before and after some optimization.
    Or the height of different populations of trees before and after some natural event.
    Distributions on the same column are slightly overlapped, to obtain a more compact layout.
    Inspired by https://seaborn.pydata.org/examples/kde_ridgeplot.html

    The total figure size is computed as follows:
    * `number_of_plots = len(data[identifier_column]).unique()`
    * `number_of_rows = ceil(number_of_plots / number_of_plot_columns)`
    * `width = number_of_plot_columns * plot_height * aspect_ratio`
    * `height = number_of_rows * plot_height`
    For example, to obtain a square figure of shape `(3.5, 3.5)`:
    * `plot_height = 3.5 / number_of_plots`
    * `aspect_ratio = number_of_plots`

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
    :param plot_height: Height of each plot.
    :param aspect_ratio: Aspect ratio of each plot.
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
    :param horizontal_spacing: Spacing between columns of the plot, as a fraction of the figure width,
        provided to `plt.subplots_adjust` as `wspace`. A value of 0 means no spacing. Applied only if `ax` is None.
    :param vertical_spacing: Spacing between rows of the plot, as a fraction of the figure height,
        provided to `plt.subplots_adjust` as `hspace`. A value of 0 means no spacing. Applied only if `ax` is None.
    :param reset_plot_style: If True, reset the style of the plot before plotting.
        Disabling it can be useful when plotting on an existing axis rather than creating a new one,
        and the existing axis has a custom style.
    :return: The Seaborn FacetGrid where the plot is contained.
    """

    ##############
    # Setup data #
    ##############

    # As we are plotting on N columns, we need to explicitely assign the column to each plot;
    row_identifier = "row_num"
    col_identifier = "col_num"
    _data = _setup_data(
        data=data,
        identifier_column=identifier_column,
        number_of_plot_columns=number_of_plot_columns,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
    )

    ##############
    # Setup plot #
    ##############

    g = _setup_plot(
        data=_data,
        identifier_column=identifier_column,
        plot_height=plot_height,
        aspect_ratio=aspect_ratio,
        vertical_line_y_max=0.8,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
        reset_plot_style=reset_plot_style,
    )
    axes: list[list[Axes]] = g.axes.tolist()

    ##################
    # Add main plots #
    ##################

    # Plot the densities. Plot them twice as the second time we plot just the black contour.
    # "cut" removes values above the threshold; clip=xlimits avoids plotting values outside the margins;
    _plot_densities(g=g, column_1=column_1, column_2=column_2, xlimits=xlimits, palette=palette, line_width=0.7)

    #####################
    # Style fine-tuning #
    #####################

    _style_fine_tuning(
        g=g,
        xlimits=xlimits,
        x_axis_line_width=0.9,
    )
    _add_individual_plot_labels(
        g=g,
        plot_label_x_position=0,
        plot_label_y_position=0.15,
        font_size=font_size * 1.2,
        identifier_column=identifier_column,
    )

    # Disable y ticks and remove axis;
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Set the ticks and tick labels. Hide all of them except for the last row of each column;
    for ax_i in axes[:-1]:
        for ax_j in ax_i:
            assert isinstance(ax_j, Axes)
            ax_j.set(xticks=[])
            ax_j.set(xticklabels=[])
    for ax_j in axes[-1]:
        assert isinstance(ax_j, Axes)
        ax_j.xaxis.set_tick_params(grid_linewidth=0)  # Hide some white markers that are still present;
        ax_j.xaxis.set_major_formatter(_x_ticks_major_formatter)
        ax_j.tick_params(axis="x", which="major", labelsize=font_size * 0.8, width=0.9, length=4, color="#2f2f2f")
        for tic in ax_j.xaxis.get_major_ticks():
            tic.tick1line.set_visible(True)
            tic.tick2line.set_visible(False)
    # Set labels on the last axis of each column (except the last, which could have fewer plots).
    # Hide labels and ticks on empty axes;
    _set_x_labels(
        xlabel=xlabel,
        data=_data,
        axes=axes,
        identifier_column=identifier_column,
        row_identifier=row_identifier,
        col_identifier=col_identifier,
        font_size=font_size,
    )

    # Fix the borders. This must be done here as the previous operations update the default values;
    plt.subplots_adjust(
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        hspace=vertical_spacing,
        wspace=horizontal_spacing,
    )

    # Add custom legend;
    _add_legend(palette=palette, legend_labels=legend_labels, g=g, font_size=font_size)

    return g
