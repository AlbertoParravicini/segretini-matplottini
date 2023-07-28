from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LinearLocator

# from segretini_matplottini.utils.plot_utils import reset_plot_style
from segretini_matplottini.utils import (
    false_negatives as _false_negatives,
)
from segretini_matplottini.utils import (
    false_positives as _false_positives,
)
from segretini_matplottini.utils import (
    reset_plot_style as _reset_plot_style,
)
from segretini_matplottini.utils import (
    true_negatives as _true_negatives,
)
from segretini_matplottini.utils import (
    true_positives as _true_positives,
)
from segretini_matplottini.utils.colors import MEGA_PINK
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE


def _plot_binary_classification_curve(
    x: Float[np.ndarray, "#n"],
    y: Float[np.ndarray, "#n"],
    xlabel: str,
    ylabel: str,
    xlimits: Optional[tuple[int, int]] = None,
    ylimits: Optional[tuple[int, int]] = None,
    annotate_best_y: bool = False,
    annotate_area_under_curve: bool = False,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    color: str = MEGA_PINK,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.95,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a curve for a binary classification problem, e.g. the True Positives, or the F1,
    for different binary classification thresholds.

    :param x: An array of values with coordinates for the x-axis.
        In most cases, these can be thresholds (e.g. for F1) or values
        for a specific metric (e.g. Recall in a Precision-Recall curve).
    :param y: An array of values with coordinates for the y-axis.
        In most cases, these are values for a specific metric (e.g. F1 for F1, or Recall in a Precision-Recall curve).
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param xlimits: Limits for the x-axis.
    :param ylimits: Limits for the y-axis.
    :param annotate_best_y: If True, add an annotation with the best value for y (e.g. the best F1 score),
        and the corresponding value for x (e.g. the threshold that maximizes F1).
    :param annotate_area_under_curve: If True, add an annotation with the area under the curve.
        Useful to measure the area under the curve for plots where
        the metric is meaningful (e.g. Precision-Recall curve).
    :param x_axis_ticks_count: Number of ticks on the x-axis.
    :param y_axis_ticks_count: Number of ticks on the y-axis.
    :param color: Color for the curve and the filling color.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is a bit less than half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two or three curves per column.
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
    # Setup plot #
    ##############

    # Deal with missing settings;
    if xlimits is None:
        xlimits = (min(x), max(x))
    if ylimits is None:
        ylimits = (0, max(y))

    # Initialize figure;
    if reset_plot_style:
        _reset_plot_style(label_pad=2, xtick_major_pad=1, ytick_major_pad=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size, dpi=DEFAULT_DPI)
        plt.subplots_adjust(top=top_padding, bottom=bottom_padding, left=left_padding, right=right_padding)
    else:
        fig = ax.get_figure()

    ##################
    # Add main plots #
    ##################

    # Draw the curve;
    plt.plot(x, y, color="#2f2f2f", linewidth=0.6)
    # Fill area under curve;
    plt.fill_between(x, y, color=color, alpha=0.6)

    #####################
    # Style fine-tuning #
    #####################

    # Set axes limits;
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # Format ticks on the x axis;
    ax.xaxis.set_major_locator(LinearLocator(x_axis_ticks_count))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.1f}")
    ax.tick_params(axis="x", labelsize=font_size * 0.8)

    # Format ticks on the y axis;
    @FuncFormatter
    def _y_ticks_major_formatter(x: float, pos: str) -> str:
        assert ylimits is not None, "❌ ylimits must be provided to create y-axis tick labels"
        if ylimits[1] - ylimits[0] <= 0.5:
            return f"{x:.2f}"
        elif ylimits[1] <= 1:
            return f"{x:.1f}"
        else:
            return f"{int(x)}"

    ax.yaxis.set_major_locator(LinearLocator(y_axis_ticks_count))
    ax.yaxis.set_major_formatter(_y_ticks_major_formatter)
    ax.tick_params(axis="y", labelsize=font_size * 0.8)
    ax.tick_params(labelcolor="#2f2f2f")

    # Set grid on y axis;
    ax.xaxis.grid(False)
    ax.yaxis.grid(linewidth=0.5)

    # Add axes labels;
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)

    # Annotate best score and threshold;
    labels: list[str] = []
    if annotate_best_y:
        best_x_index = np.argmax(y)
        best_x = float(x[best_x_index])
        best_score = float(y[best_x_index])
        ax.axhline(
            y=best_score,
            xmin=xlimits[0],
            xmax=best_x,
            color=color,
            lw=0.7,
            linestyle="-",
        )
        ax.axvline(
            x=best_x,
            ymin=ylimits[0],
            ymax=best_score,
            color=color,
            lw=0.7,
            linestyle="-",
        )
        labels += [f"Best {ylabel}={best_score:.2f}", f"Best {xlabel}={best_x:.2f}"]

    # Compute the area under the curve, and annotate the plot;
    if annotate_area_under_curve:
        auc = np.trapz(y, x=x)
        labels += [f"AUC={auc:.2f}"]
    # Write labels;
    label = "\n".join(labels)
    ax.annotate(
        label,
        size=font_size * 0.8,
        xy=(0.02, 0.02),
        ha="left",
        va="bottom",
        xycoords="axes fraction",
        color="#2f2f2f",
    )
    return fig, ax


def true_positives(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "TP",
    xlimits: tuple[int, int] = (0, 1),
    color: str = MEGA_PINK,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.95,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> Axis:
    """
    Plot the True Positive curve for a binary classification problem,
    where True Positives are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
    :param num_thresholds: Number of thresholds for which the curve is computed.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param xlimits: Limits for the x-axis.
    :param color: Color for the curve and the filling color.
    :param x_axis_ticks_count: Number of ticks on the x-axis.
    :param y_axis_ticks_count: Number of ticks on the y-axis.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is a bit less than half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two or three curves per column.
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
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = [_true_positives(logits, targets, t) for t in thresholds]
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        color=color,
        annotate_area_under_curve=False,
        annotate_best_y=False,
        x_axis_ticks_count=x_axis_ticks_count,
        y_axis_ticks_count=y_axis_ticks_count,
        ax=ax,
        figure_size=figure_size,
        font_size=font_size,
        left_padding=left_padding,
        right_padding=right_padding,
        bottom_padding=bottom_padding,
        top_padding=top_padding,
        reset_plot_style=reset_plot_style,
    )


def true_negatives(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "TN",
    xlimits: tuple[int, int] = (0, 1),
    color: str = MEGA_PINK,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.95,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> Axis:
    """
    Plot the True Negatives curve for a binary classification problem,
    where True Negatives are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
    :param num_thresholds: Number of thresholds for which the curve is computed.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param xlimits: Limits for the x-axis.
    :param color: Color for the curve and the filling color.
    :param x_axis_ticks_count: Number of ticks on the x-axis.
    :param y_axis_ticks_count: Number of ticks on the y-axis.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is a bit less than half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two or three curves per column.
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
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = [_true_negatives(logits, targets, t) for t in thresholds]
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        color=color,
        annotate_area_under_curve=False,
        annotate_best_y=False,
        x_axis_ticks_count=x_axis_ticks_count,
        y_axis_ticks_count=y_axis_ticks_count,
        ax=ax,
        figure_size=figure_size,
        font_size=font_size,
        left_padding=left_padding,
        right_padding=right_padding,
        bottom_padding=bottom_padding,
        top_padding=top_padding,
        reset_plot_style=reset_plot_style,
    )


def false_positives(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "FP",
    xlimits: tuple[int, int] = (0, 1),
    color: str = MEGA_PINK,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.95,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> Axis:
    """
    Plot the False Positives curve for a binary classification problem,
    where False Positives are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
    :param num_thresholds: Number of thresholds for which the curve is computed.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param xlimits: Limits for the x-axis.
    :param color: Color for the curve and the filling color.
    :param x_axis_ticks_count: Number of ticks on the x-axis.
    :param y_axis_ticks_count: Number of ticks on the y-axis.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is a bit less than half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two or three curves per column.
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
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = [_false_positives(logits, targets, t) for t in thresholds]
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        color=color,
        annotate_area_under_curve=False,
        annotate_best_y=False,
        x_axis_ticks_count=x_axis_ticks_count,
        y_axis_ticks_count=y_axis_ticks_count,
        ax=ax,
        figure_size=figure_size,
        font_size=font_size,
        left_padding=left_padding,
        right_padding=right_padding,
        bottom_padding=bottom_padding,
        top_padding=top_padding,
        reset_plot_style=reset_plot_style,
    )


def false_negatives(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "FN",
    xlimits: tuple[int, int] = (0, 1),
    color: str = MEGA_PINK,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    ax: Optional[Axes] = None,
    figure_size: tuple[float, float] = (1.5, 1.4),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.25,
    right_padding: float = 0.95,
    bottom_padding: float = 0.2,
    top_padding: float = 0.95,
    reset_plot_style: bool = True,
) -> Axis:
    """
    Plot the False Negatives curve for a binary classification problem,
    where False Negatives are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
    :param num_thresholds: Number of thresholds for which the curve is computed.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param xlimits: Limits for the x-axis.
    :param color: Color for the curve and the filling color.
    :param x_axis_ticks_count: Number of ticks on the x-axis.
    :param y_axis_ticks_count: Number of ticks on the y-axis.
    :param ax: An axis to use for the plot. If None, create a new one.
    :param figure_size: Size of the figure, in inches. The default is a bit less than half the `\columnwidth` of
        a two-columns template in LaTeX, so that one can add two or three curves per column.
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
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = [_false_negatives(logits, targets, t) for t in thresholds]
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        color=color,
        annotate_area_under_curve=False,
        annotate_best_y=False,
        x_axis_ticks_count=x_axis_ticks_count,
        y_axis_ticks_count=y_axis_ticks_count,
        ax=ax,
        figure_size=figure_size,
        font_size=font_size,
        left_padding=left_padding,
        right_padding=right_padding,
        bottom_padding=bottom_padding,
        top_padding=top_padding,
        reset_plot_style=reset_plot_style,
    )
