from typing import Literal, Optional, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator, SecondLocator
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator

from segretini_matplottini.utils import activate_dark_background
from segretini_matplottini.utils import reset_plot_style as _reset_plot_style
from segretini_matplottini.utils.colors import MEGA_PINK
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE


def timeseries(
    x: pd.Series,
    line_color: str = MEGA_PINK,
    line_width: float = 0.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlimits: Optional[tuple[float, float]] = None,
    ylimits: Optional[tuple[float, float]] = None,
    date_format: Optional[str] = None,
    seconds_interval_major_ticks: Optional[int] = None,
    minutes_interval_major_ticks: Optional[int] = None,
    hours_interval_major_ticks: Optional[int] = None,
    seconds_interval_minor_ticks: Optional[int] = None,
    minutes_interval_minor_ticks: Optional[int] = None,
    hours_interval_minor_ticks: Optional[int] = None,
    draw_style: Literal["default", "steps-pre", "steps-mid", "steps-post", "stem"] = "default",
    fill: bool = False,
    dark_background: bool = False,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (9, 3),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.07,
    right_padding: float = 0.98,
    bottom_padding: float = 0.23,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot an array of numerical data, representing a time-series of some kind.

    :param x: A Series of numerical data, where the index can be a timestamp.
    :param line_color: Color of the time-series line.
    :param line_width: Width of the time-series line.
    :param xlabel: Label of the x-axis.
    :param ylabel: Label of the y-axis.
    :param xlimits: Limits of the y-axis. If none, use `[min(x), max(x)]`.
    :param ylimits: Limits of the y-axis. If none, use `[min(x), max(x)]`.
    :param date_format: If not None, try formatting x-axis tick labels with the specified time format.
    :param seconds_interval_major_ticks: If not None and `date_format` is present,
        locate ticks on the major axis with distance equal to this amount of seconds.
        If `minutes_interval` or `hours_interval` are also present, use the smallest interval possible.
        If none of them is present, locate ticks every 1 minute.
    :param minutes_interval_major_ticks: If not None and `date_format` is present,
        locate ticks on the major axis with distance equal to this amount of minutes.
    :param hours_interval_major_ticks: If not None and `date_format` is present,
        locate ticks on the major axis with distance equal to this amount of hours.
    :param seconds_interval_minor_ticks: If not None and `date_format` is present,
        locate ticks on the minor axis with distance equal to this amount of seconds.
        If `minutes_interval` or `hours_interval` are also present, use the smallest interval possible.
        If none of them is present, locate ticks every 1 minute.
    :param minutes_interval_minor_ticks: If not None and `date_format` is present,
        locate ticks on the minor axis with distance equal to this amount of minutes.
    :param hours_interval_minor_ticks: If not None and `date_format` is present,
        locate ticks on the minor axis with distance equal to this amount of hours.
    :param draw_style: Style of the line, as in Matplotlib's `draw_style`. For `default`, the points are connected with straight lines.
        Alternatively, connect points with steps, or plot a stem plot.
        `steps-pre`: The step is at the beginning of the line segment, i.e. the line will be at the y-value of point to the right.
        `steps-mid`: The step is halfway between the points.
        `steps-post`: The step is at the end of the line segment, i.e. the line will be at the y-value of the point to the left.
        `stem`: Plot a stem plot, i.e. a vertical line at each x location from the baseline to y.
    :param fill: If True, fill the area under the line. Ignored if `draw_style` is `stem`.
    :param dark_background: If True, plot on a dark background.
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
    :return: Matplotlib figure and axis containing the plot.
    """

    ##############
    # Setup data #
    ##############

    y = np.array(x.copy())
    try:
        x = np.array(pd.to_datetime(x.index))
    except ValueError:
        x = np.arange(len(x))

    ##############
    # Setup plot #
    ##############

    if reset_plot_style:
        _reset_plot_style(label_pad=4, xtick_major_pad=3, ytick_major_pad=3)
    if dark_background:
        activate_dark_background()
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

    if draw_style == "stem":
        stems = ax.stem(x, y, linefmt=line_color, markerfmt=" ", basefmt=" ")
        plt.setp(stems, "linewidth", line_width)  # Set stem line width
    else:
        ax.plot(x, y, lw=line_width, color=line_color, drawstyle=draw_style)
        if fill:
            step = None if "steps" not in draw_style else draw_style.replace("steps-", "")
            assert step is None or step in get_args(
                Literal["pre", "mid", "post"]
            ), f"❌ invalid step value, must be 'pre', 'mid' or 'post', not {step}"
            ax.fill_between(x, y, alpha=0.5, color=line_color, step=step)  # type: ignore

    #####################
    # Style fine-tuning #
    #####################

    # Activate grid on the y axis
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    # Set axes limits
    if xlimits is None:
        ax.set_xlim(min(x), max(x))
    else:
        ax.set_xlim(xlimits)
    if ylimits is None:
        ax.set_ylim(min(y), max(y))
    else:
        ax.set_ylim(ylimits)

    # Format x-axis tick labels as a date
    if date_format is not None:
        # Major ticks
        if seconds_interval_major_ticks is not None:
            ax.xaxis.set_major_locator(SecondLocator(interval=seconds_interval_major_ticks))
        elif minutes_interval_major_ticks is not None:
            ax.xaxis.set_major_locator(MinuteLocator(interval=minutes_interval_major_ticks))
        elif hours_interval_major_ticks is not None:
            ax.xaxis.set_major_locator(HourLocator(interval=hours_interval_major_ticks))
        else:
            ax.xaxis.set_major_locator(MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        # Minor ticks
        if seconds_interval_minor_ticks is not None:
            ax.xaxis.set_minor_locator(SecondLocator(interval=seconds_interval_minor_ticks))
        elif minutes_interval_minor_ticks is not None:
            ax.xaxis.set_minor_locator(MinuteLocator(interval=minutes_interval_minor_ticks))
        elif hours_interval_minor_ticks is not None:
            ax.xaxis.set_minor_locator(HourLocator(interval=hours_interval_minor_ticks))
        ax.tick_params(axis="x", which="major", labelsize=font_size - 2, rotation=45)
        plt.xticks(ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(axis="x", which="major", labelsize=font_size)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.3f}")
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.tick_params(axis="y", which="major", labelsize=font_size - 2)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size)

    return fig, ax
