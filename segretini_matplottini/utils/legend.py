from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import allow_rasterization
from matplotlib.axis import Axis
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch, Shadow


class LegendWithDarkShadow(Legend):
    """
    A custom legend style with a rectangular box and a dark shadow around the box.
    """

    def __post_init__(self) -> None:
        self.shadow_offset = 2

    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        # docstring inherited
        if not self.get_visible():
            return

        renderer.open_group("legend", gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # If mode == fill, set the width of the legend_box to the
        # width of the parent (minus pads);
        if self._mode in ["expand"]:
            pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
            self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

        # Update the location and size of the legend. This needs to
        # be done in any case to clip the figure right;
        bbox = self._legend_box.get_window_extent(renderer)
        self.legendPatch.set_bounds(bbox.x0, bbox.y0, bbox.width, bbox.height)
        self.legendPatch.set_mutation_scale(fontsize)

        if self.shadow:
            Shadow(self.legendPatch, self.shadow_offset, -self.shadow_offset, alpha=1).draw(renderer)

        self.legendPatch.draw(renderer)
        self._legend_box.draw(renderer)

        renderer.close_group("legend")
        self.stale = False


def add_legend_with_dark_shadow(
    handles: list[Patch],
    labels: list[str],
    fig: Optional[Figure] = None,
    ax: Optional[Axis] = None,
    shadow_offset: int = 2,
    line_width: float = 0.5,
    *args: Any,
    **kwargs: Any,
) -> tuple[LegendWithDarkShadow, Axis]:
    """
    Add a legend with dark shadow to the specified axis.

    :param fig: Figure where the legend is added.
    :param ax: Axis where the legend is added. If both figure and axis are specified,
        draw the legend in the axis. If neither figure nor axis are specified, draw the legend in the current figure.
    :param shadow_offset: Offset of the shadow from the legend.
    :param line_width: Width of the legend box.
    :param args: Any argument passed to the legend constructor.
    :param kwargs: Any keyword argument passed to the legend constructor.
    """
    # If both figure and axis are missing, draw in the current figure;
    if fig is None and ax is None:
        fig = plt.gcf()
    legend_kwargs = (
        dict(
            fancybox=False,
            framealpha=1,
            shadow=True,
            edgecolor="#2f2f2f",
        )
        | kwargs
    )
    # If the axis is specified, always draw in the axis;
    where_to_draw = fig
    if ax is not None:
        # Draw in the current axis;
        handles, labels, _, kwargs = matplotlib.legend._parse_legend_args(
            [ax], handles=handles, labels=labels, *args, **kwargs
        )
        where_to_draw = ax
    leg = LegendWithDarkShadow(
        where_to_draw,
        handles,
        labels,
        **legend_kwargs,
    )
    leg.shadow_offset = shadow_offset
    leg.get_frame().set_linewidth(line_width)
    # Add legend to the axis;
    if ax is None:
        ax = plt.gca()
    ax.legend_ = leg
    return leg, ax


def transpose_legend_labels(
    labels: list[str], handles: list[Patch], max_elements_per_row: int = 6, default_elements_per_col: int = 2
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
