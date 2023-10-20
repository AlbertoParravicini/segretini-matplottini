from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch, Rectangle


def get_legend_handles_from_colors(
    colors: list[str], edge_color: str = "#2f2f2f", line_width: float = 0.5
) -> list[Rectangle]:
    """
    Obtain a list of handles to create a legend from a list of colors.

    :param colors: List of colors.
    :param edge_color: Color of the edge of the legend handle.
    :param line_width: Width of the line that surrounds the handle.
    :return: List of handles.
    """
    return [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=c,
            edgecolor=edge_color,
            linewidth=line_width,
        )
        for c in colors
    ]


def add_legend_with_dark_shadow(
    handles: Sequence[Artist],
    labels: list[str],
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    shadow_offset: int = 2,
    line_width: float = 0.5,
    **kwargs: Any,
) -> tuple[Legend, Axes]:
    """
    Add a legend with dark shadow to the specified axis.

    :param fig: Figure where the legend is added.
    :param ax: Axes where the legend is added. If both figure and axis are specified,
        draw the legend in the axis. If neither figure nor axis are specified, draw the legend in the current figure.
    :param shadow_offset: Offset of the shadow from the legend.
    :param line_width: Width of the legend box.
    :param kwargs: Any keyword argument passed to the legend constructor.
    """
    # If both figure and axis are missing, draw in the current figure.
    # If the axis is specified, always draw in the axis
    _fig: Figure = fig if fig is not None else plt.gcf()
    where_to_draw: Union[Figure, Axes] = _fig
    if ax is not None:
        where_to_draw = ax
    legend_kwargs: Any = dict(fancybox=False, framealpha=1, edgecolor="#2f2f2f") | kwargs
    leg = Legend(
        where_to_draw,
        handles,
        labels,
        shadow=dict(
            ox=shadow_offset,
            oy=-shadow_offset,
            alpha=1,
        ),
        **legend_kwargs,
    )
    leg.get_frame().set_linewidth(line_width)
    # Add legend to the axis;
    if ax is None:
        ax = plt.gca()
    ax.legend_ = leg
    return leg, ax


def transpose_legend_labels(
    labels: list[str],
    handles: list[Patch],
    max_elements_per_row: int = 6,
    default_elements_per_col: int = 2,
) -> tuple[list[str], list[Patch]]:
    """
    Matplotlib by defaults places elements in the legend from top to bottom.
    In most cases, placing them left-to-right is more readable (English is read left-to-right, not top-to-bottom)
    This function transposes the elements in the legend,
    allowing to set the maximum number of values you want in each row.

    :param labels: List of textual labels in the legend.
    :param handles: List of color patches in the legend.
    :param max_elements_per_row: Maximum number of legend elements per row.
    :param default_elements_per_col: By default, try having default_elements_per_col elements
        in each column (could be more if max_elements_per_row is reached).
    """
    elements_per_row = min(
        int(np.ceil(len(labels) / default_elements_per_col)), max_elements_per_row
    )  # Don't add too many elements per row;
    labels = np.concatenate([labels[i::elements_per_row] for i in range(elements_per_row)], axis=0).tolist()
    handles = np.concatenate([handles[i::elements_per_row] for i in range(elements_per_row)], axis=0).tolist()
    return labels, handles
