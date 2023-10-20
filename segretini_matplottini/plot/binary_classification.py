from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float, Integer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LinearLocator
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from segretini_matplottini.utils import adjust_rows_and_columns_to_number_of_plots

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
        _fig = ax.get_figure()
        assert _fig is not None, "❌ the axis has no figure associated"
        fig = _fig

    ##################
    # Add main plots #
    ##################

    # Draw the curve;
    ax.plot(x, y, color="#2f2f2f", linewidth=0.5, zorder=3)
    # Fill area under curve;
    ax.fill_between(x, y, color=color, alpha=0.8, zorder=2)

    #####################
    # Style fine-tuning #
    #####################

    # Set axes limits;
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # Format ticks on the x-axis;
    ax.xaxis.set_major_locator(LinearLocator(x_axis_ticks_count))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.1f}")
    ax.tick_params(axis="x", labelsize=font_size * 0.7)

    # Format ticks on the y-axis;
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
    ax.tick_params(axis="y", labelsize=font_size * 0.7)
    ax.tick_params(labelcolor="#2f2f2f")

    # Set grid on y-axis;
    ax.grid(axis="y", linewidth=0.5, linestyle="--")

    # Reduce the size of the ticks;
    ax.tick_params(axis="both", which="major", length=2)

    # Add axes labels;
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    # Annotate best score and threshold;
    labels: list[str] = []
    if annotate_best_y:
        best_x_index = np.argmax(y)
        best_x = float(x[best_x_index])
        best_score = float(y[best_x_index])
        ax.axhline(y=best_score, xmin=xlimits[0], xmax=best_x, color=color, lw=0.4, linestyle="-", zorder=1)
        ax.axvline(x=best_x, ymin=ylimits[0], ymax=best_score, color=color, lw=0.4, linestyle="-", zorder=1)
        labels += [f"Best {ylabel}={best_score:.2f}", f"Best {xlabel}={best_x:.2f}"]

    # Compute the area under the curve, and annotate the plot;
    if annotate_area_under_curve:
        auc = np.trapz(y, x=x)
        labels += [f"AUC={auc:.2f}"]
    # Write labels;
    label = "\n".join(labels)
    ax.annotate(
        label,
        size=font_size * 0.7,
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
) -> tuple[Figure, Axes]:
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
    scores = np.array([_true_positives(logits, targets, t) for t in thresholds], dtype=float)
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
) -> tuple[Figure, Axes]:
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
    scores = np.array([_true_negatives(logits, targets, t) for t in thresholds], dtype=float)
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
) -> tuple[Figure, Axes]:
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
    scores = np.array([_false_positives(logits, targets, t) for t in thresholds], dtype=float)
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
) -> tuple[Figure, Axes]:
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
    scores = np.array([_false_negatives(logits, targets, t) for t in thresholds], dtype=float)
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


def precision(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "Precision",
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
) -> tuple[Figure, Axes]:
    """
    Plot the Precision curve for a binary classification problem,
    where Precision are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    From `sklearn` documentation:
    The Recall is the ratio `TP / (TP + FP)` where `TP` is the number of True Positives and `FP` the number
    of False Negatives.  The Precision is intuitively the ability of the classifier
    not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.

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
    scores = np.array([precision_score(y_pred=logits >= t, y_true=targets, zero_division=0) for t in thresholds])
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        ylimits=(0, 1),
        color=color,
        annotate_area_under_curve=True,
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


def recall(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "Recall",
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
) -> tuple[Figure, Axes]:
    """
    Plot the Recall curve for a binary classification problem,
    where Recall are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    From `sklearn` documentation:
    The Recall is the ratio `TP / (TP + FN)` where `TP` is the number of True Positives and `FN` the number
    of False Negatives. The Recall is intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.

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
    scores = np.array([recall_score(y_pred=logits >= t, y_true=targets, zero_division=1) for t in thresholds])
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        ylimits=(0, 1),
        color=color,
        annotate_area_under_curve=True,
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


def f1(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "Threshold",
    ylabel: str = "F1",
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
) -> tuple[Figure, Axes]:
    """
    Plot the F1 curve for a binary classification problem,
    where F1 are computed for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.

    From `sklearn` documentation:
    The F1 score can be interpreted as a harmonic mean of the Precision and Recall,
    where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of Precision and Recall to the F1 score are equal.
    The formula for the F1 score is: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

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
    scores = np.array([f1_score(y_pred=logits >= t, y_true=targets, zero_division=0) for t in thresholds])
    return _plot_binary_classification_curve(
        x=thresholds,
        y=scores,
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        ylimits=(0, 1),
        color=color,
        annotate_area_under_curve=True,
        annotate_best_y=True,
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


def roc(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    num_thresholds: int = 100,
    xlabel: str = "FP Rate",
    ylabel: str = "TP Rate",
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
) -> tuple[Figure, Axes]:
    """
    Plot the ROC curve for a binary classification problem, showing the tradeoff
    of False Positive Rate and True Positive rate for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.
    True Positive Rate is also known as Recall or Sensitivity, computed as `TP / (TP + FN)`.
    False Positive Rate is computed as `FP / (FP + TN)`.

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
    tpr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds:
        p: Bool[np.ndarray, "#n"] = (
            (logits >= threshold).astype(bool) if threshold < 1 else np.zeros_like(logits).astype(bool)
        )
        t: Bool[np.ndarray, "#n"] = targets.astype(bool)
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        fn = (~p & t).sum()
        tn = (~p & ~t).sum()
        tpr += [tp / (tp + fn)]
        fpr += [fp / (fp + tn)]
    return _plot_binary_classification_curve(
        x=np.array(fpr[::-1]),
        y=np.array(tpr[::-1]),
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        ylimits=(0, 1),
        color=color,
        annotate_area_under_curve=True,
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


def precision_recall(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    xlabel: str = "Recall",
    ylabel: str = "Precision",
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
) -> tuple[Figure, Axes]:
    """
    Plot the Precision curve for a binary classification problem, showing the tradeoff
    of Precision and Recall rate for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.
    Precision is as `TP / (TP + FP)`, Recall is computed as `TP / (TP + FN)`.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
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
    (
        precisions,
        recalls,
        _,
    ) = precision_recall_curve(targets, logits)
    # Reverse the points, and remove the last one since was added by sklearn
    # to have the plot end at (0, 0), but that's something we enforce anyway.
    return _plot_binary_classification_curve(
        x=np.array(recalls[:-1][::-1]),
        y=np.array(precisions[:-1][::-1]),
        xlabel=xlabel,
        ylabel=ylabel,
        xlimits=xlimits,
        ylimits=(0, 1),
        color=color,
        annotate_area_under_curve=True,
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


def binary_classification(
    logits: Float[np.ndarray, "#n"],
    targets: Integer[np.ndarray, "#n"],
    plot_true_positives: bool = True,
    plot_false_negatives: bool = True,
    plot_false_positives: bool = True,
    plot_true_negatives: bool = True,
    plot_precision: bool = True,
    plot_recall: bool = True,
    plot_roc: bool = True,
    plot_f1: bool = True,
    plot_precision_recall: bool = True,
    number_of_rows: Optional[int] = None,
    number_of_columns: Optional[int] = None,
    num_thresholds: int = 100,
    color: str = MEGA_PINK,
    x_axis_ticks_count: int = 6,
    y_axis_ticks_count: int = 6,
    figure_size: tuple[float, float] = (3.5, 3.2),
    font_size: int = DEFAULT_FONT_SIZE,
    left_padding: float = 0.09,
    right_padding: float = 0.98,
    bottom_padding: float = 0.08,
    top_padding: float = 0.98,
    horizontal_spacing: float = 0.5,
    vertical_spacing: float = 0.4,
    reset_plot_style: bool = True,
) -> tuple[Figure, list[list[Axes]]]:
    """
    Plot the Precision curve for a binary classification problem, showing the tradeoff
    of Precision and Recall rate for different classification thresholds from 0 to 1.
    The input is an array of numerical logits from 0 to 1,
    and an array of integer binary targets (0 or 1). The two arrays must have the same length.
    Precision is as `TP / (TP + FP)`, Recall is computed as `TP / (TP + FN)`.

    :param logits: An array of logits, with values between 0 and 1.
    :param targets: An array of targets, with values 0 or 1.
    :param plot_true_positives: If True, plot the number of True Positives for different thresholds.
    :param plot_false_negatives: If True, plot the number of False Negatives for different thresholds.
    :param plot_false_positives: If True, plot the number of False Positives for different thresholds.
    :param plot_true_negatives: If True, plot the number of True Negatives for different thresholds.
    :param plot_precision: If True, plot the Precision for different thresholds.
    :param plot_recall: If True, plot the Recall for different thresholds.
    :param plot_roc: If True, plot the ROC curve (TP Rate vs. FP Rate curve) for different thresholds,
        and the corresponding AUC value.
    :param plot_f1: If True, plot the F1 score for different thresholds.
    :param plot_precision_recall: If True, plot the Precision-Recall curve.
    :param num_thresholds: Number of thresholds for which the curves are computed.
        More thresholds means a smoother curve, but more computation time.
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

    # Specify the order in which plots have to appear, if they are enabled;
    plotting_functions: list[Callable] = []
    if plot_true_positives:
        plotting_functions.append(true_positives)
    if plot_false_negatives:
        plotting_functions.append(false_negatives)
    if plot_precision:
        plotting_functions.append(precision)
    if plot_false_positives:
        plotting_functions.append(false_positives)
    if plot_true_negatives:
        plotting_functions.append(true_negatives)
    if plot_recall:
        plotting_functions.append(recall)
    if plot_roc:
        plotting_functions.append(roc)
    if plot_f1:
        plotting_functions.append(f1)
    if plot_precision_recall:
        plotting_functions.append(precision_recall)

    # Compute the number of rows and columns
    # Obtain the number of rows and columns to plot;
    _number_of_rows, _number_of_columns = adjust_rows_and_columns_to_number_of_plots(
        number_of_rows=number_of_rows,
        number_of_columns=number_of_columns,
        number_of_plots=len(plotting_functions),
    )

    if reset_plot_style:
        _reset_plot_style(label_pad=1, xtick_major_pad=1, ytick_major_pad=1, border_width=0.6)
    fig, axes = plt.subplots(_number_of_rows, _number_of_columns, figsize=figure_size, dpi=DEFAULT_DPI, squeeze=False)
    plt.subplots_adjust(
        left=left_padding,
        right=right_padding,
        bottom=bottom_padding,
        top=top_padding,
        wspace=horizontal_spacing,
        hspace=vertical_spacing,
    )
    for i, plotting_function in enumerate(plotting_functions):
        plotting_function_kwargs: dict[str, Any] = dict(
            logits=logits,
            targets=targets,
            ax=axes.flat[i],
            color=color,
            font_size=font_size,
            num_thresholds=num_thresholds,
            x_axis_ticks_count=x_axis_ticks_count,
            y_axis_ticks_count=y_axis_ticks_count,
        )
        # Remove a parameter not supported by `precision_recall`
        if plotting_function == precision_recall:
            del plotting_function_kwargs["num_thresholds"]
        plotting_function(**plotting_function_kwargs)

    # Delete the extra plots
    for ax in axes.flat[len(plotting_functions) :]:
        ax.remove()

    # Convert the axes array to a 2D list.
    # Remove deleted axes, by checking if they no longer have a figure reference
    axes_list: list[list[Axes]] = axes.tolist()
    non_stale_axes_list: list[list[Axes]] = [[ax_j for ax_j in ax_i if ax_j.figure is not None] for ax_i in axes_list]
    return fig, non_stale_axes_list
