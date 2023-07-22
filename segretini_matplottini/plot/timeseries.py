from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator, SecondLocator
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator

from segretini_matplottini.utils.plot_utils import reset_plot_style


def timeseries(
    x: pd.Series,
    line_color: str = "#FF6494",
    line_width: float = 0.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylimits: Optional[tuple[float, float]] = None,
    aspect_ratio: float = 3,
    date_format: Optional[str] = None,
    seconds_interval: Optional[int] = None,
    minutes_interval: Optional[int] = None,
    hours_interval: Optional[int] = None,
    font_size: int = 10,
    draw_style: str = "default",
    fill: bool = False,
    dark_background: bool = False,
) -> tuple[Figure, Axis]:
    """
    Plot an array of numerical data, representing a time-series of some kind.

    :param x: A Series of numerical data, where the index can be a timestamp.
    :param line_color: Color of the time-series line.
    :param line_width: Width of the time-series line.
    :param xlabel: Label of the x-axis.
    :param ylabel: Label of the y-axis.
    :param ylimits: Limits of the y-axis. If none, use `[min(x), max(x)]`.
    :param aspect_ratio: Aspect ratio of the plot. The size is computed as `[aspect_ration * 3, 3]`.
    :param date_format: If not None, try formatting x-axis tick labels with the specified time format.
    :param seconds_interval: If not None and `date_format` is present, locate ticks with distance equal to this amount of seconds.
        If `minutes_interval` or `hours_interval` are also present, use the smallest interval possible.
        If none of them is present, locate ticks every 1 minute.
    :param minutes_interval: If not None and `date_format` is present, locate ticks with distance equal to this amount of minutes.
    :param hours_interval: If not None and `date_format` is present, locate ticks with distance equal to this amount of hours.
    :param font_size: Default font size used in the plot.
    :param draw_style: Style of the line, as in Matplotlib's `draw_style`. For `default`, the points are connected with straight lines.
        Alternatively, connect points with steps.
        `steps-pre`: The step is at the beginning of the line segment, i.e. the line will be at the y-value of point to the right.
        `steps-mid`: The step is halfway between the points.
        `steps-post`: The step is at the end of the line segment, i.e. the line will be at the y-value of the point to the left.
    :param dark_background: If True, plot on a dark background.
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

    reset_plot_style(label_pad=4, xtick_major_pad=3, ytick_major_pad=3, dark_background=dark_background)
    fig, ax = plt.subplots(figsize=(aspect_ratio * 3, 3), dpi=600)
    plt.subplots_adjust(top=0.95, bottom=0.25, left=0.08, right=0.98)

    ##################
    # Add main plots #
    ##################

    ax.plot(x, y, lw=line_width, color=line_color, drawstyle=draw_style)
    if fill:
        step = None if "steps" not in draw_style else draw_style.replace("steps-", "")
        plt.fill_between(x, y, alpha=0.5, color=line_color, step=step)

    #####################
    # Style fine-tuning #
    #####################

    # Activate grid on the y axis
    ax.grid(True, axis="y", lw=0.8)
    ax.grid(False, axis="x")

    # Set axes limits
    ax.set_xlim(min(x), max(x))
    if ylimits is None:
        ax.set_ylim(min(y), max(y))
    else:
        ax.set_ylim(ylimits)

    # Format x-axis tick labels as a date
    if date_format is not None:
        if seconds_interval is not None:
            ax.xaxis.set_major_locator(SecondLocator(interval=seconds_interval))
        elif minutes_interval is not None:
            ax.xaxis.set_major_locator(MinuteLocator(interval=minutes_interval))
        elif hours_interval is not None:
            ax.xaxis.set_major_locator(HourLocator(interval=hours_interval))
        else:
            ax.xaxis.set_major_locator(MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        ax.tick_params(axis="x", which="major", labelsize=font_size - 2, rotation=45)
        plt.xticks(ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(axis="x", which="major", labelsize=font_size)
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.tick_params(axis="y", which="major", labelsize=font_size - 2)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size)

    return fig, ax
