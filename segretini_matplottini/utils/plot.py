from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from segretini_matplottini.utils.colors import BACKGROUND_BLACK
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE


def activate_dark_background(background_color: str = BACKGROUND_BLACK) -> None:
    """
    Modify the current plot style to use a dark background,
    and to save figures with the dark background.

    :param background_color: Hexadecimal color used for the background.
    """
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = background_color
    plt.rcParams["savefig.facecolor"] = background_color


def reset_plot_style(
    label_pad: float = 0,
    xtick_major_pad: float = 1,
    ytick_major_pad: float = 1,
    border_width: float = 0.8,
    grid_linewidth: Optional[float] = None,
    title_size: Optional[float] = None,
    title_pad: Optional[float] = None,
    label_size: Optional[float] = None,
    dark_background: bool = False,
) -> None:
    """
    Initialize the plot with a consistent style.

    :param label_pad: Padding between axis and axis label.
    :param xtick_major_pad: Padding between axis ticks and tick labels.
    :param ytick_major_pad: Padding between axis ticks and tick labels.
    :param grid_linewidth: Width of lines in the grid, when enabled.
    :param title_size: Size of the title.
    :param title_pad: Padding of the title.
    :param label_size: Size of the labels.
    :param border_width: Line width of the axis borders.
    :param dark_background: If True, use a dark background.
    """
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Setup general plotting settings;
    style_dict: dict[str, Any] = {"ytick.left": True, "xtick.bottom": True}
    if grid_linewidth is not None:
        # Turn on the grid for the y-axis
        style_dict["axes.grid"] = True
        style_dict["axes.grid.axis"] = "y"
        style_dict["grid_linewidth"] = grid_linewidth
    sns.set_style("white", style_dict)
    # Other parameters
    plt.rcParams["axes.labelpad"] = label_pad
    plt.rcParams["xtick.major.pad"] = xtick_major_pad
    plt.rcParams["ytick.major.pad"] = ytick_major_pad
    plt.rcParams["axes.linewidth"] = border_width
    if title_size is not None:
        plt.rcParams["axes.titlesize"] = title_size
    if title_pad is not None:
        plt.rcParams["axes.titlepad"] = title_pad
    if label_size is not None:
        plt.rcParams["axes.labelsize"] = label_size
    # Background color
    if dark_background:
        activate_dark_background()


def add_arrow_to_barplot(
    ax: Axes,
    higher_is_better: bool = True,
    line_width: float = 0.5,
    left_margin_to_add: float = 0.1,
    arrow_color: str = "#2f2f2f",
) -> Axes:
    """
    Add a arrow before the first bar in the barplot, to indicate that higher is better,
    or that lower is better. Add a bit of space to the left, to make space for the arrow.

    :param: ax: The axis containing the barplot.
    :param: higher_is_better: If True, add an arrow that points up, to indicate that higher is better.
        If False, add an arrow that points down, to indicate that lower is better.
    :param line_width: Width of the arrow line.
    :param left_margin_to_add: Amount of space, in inches, to add to the left of the first bar,
        to make space for the arrow.
    :param arrow_color: Color of the arrow, as hexadecimal string.
    :return: The axis containing the barplot, with the arrow added.
    """
    # Add a bit of whitespace before the first bar;
    ax.set_xlim(ax.get_xlim()[0] - left_margin_to_add, ax.get_xlim()[1])
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) > 0, "❌ no bars found in the plot, make sure to draw a barplot first!"
    x_coord = rectangles[0].get_x() - (rectangles[0].get_x() - ax.get_xlim()[0]) / 2
    ax.annotate(
        "",
        xy=(x_coord, ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        xytext=(x_coord, ax.get_ylim()[0] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        arrowprops=dict(
            arrowstyle="->" if higher_is_better else "<-",
            color=arrow_color,
            linewidth=line_width,
        ),
        annotation_clip=False,
    )
    return ax


def get_exp_label(
    value: float,
    prefix: str = "",
    integer_mantissa: bool = True,
    decimal_places: int = 2,
    skip_mantissa_if_equal_to_one: bool = True,
) -> str:
    """
    Format a label in scientific notation, using Latex math font.
    For example, 10000 -> "10^4".

    :param value: Numeric label to format.
    :param prefix: String prefix added in front of the formatted label, e.g. "Time = ".
    :param integer_mantissa: If True, return a label whose mantissa is an integer number. For example,
        with `get_exp_label(0.00123, integer_mantissa=False) -> $\\mathdefault{1.23\\!·\\!{10}^{-3}}$`,
        while `get_exp_label(0.00123, integer_mantissa=True) -> $\\mathdefault{123\\!·\\!{10}^{-1}}$`.
    :param decimal_places: Number of digits to have in the decimal places, if not using an integer mantissa.
    :param skip_mantissa_if_equal_to_one: Do not add the mantissa if it is equal to 1.
    :return: Label formatted in scientific notation.
    """
    string = "{:.{prec}E}".format(value, prec=decimal_places)
    decimal_part = float(string.split("E")[0])
    sign = string.split("E")[1][0]
    exponent = int(string.split("E")[1][1:])
    if integer_mantissa:
        while (decimal_part - int(decimal_part) > 0) if value > 0 else (decimal_part - int(decimal_part) < 0):
            decimal_part *= 10
            decimal_part = float("{:.{prec}f}".format(decimal_part, prec=decimal_places))
            exponent -= 1
        decimal_part = int(decimal_part)
    separator = r"\!·\!"
    decimal_part_str = str(decimal_part)
    if skip_mantissa_if_equal_to_one and decimal_part == 1:
        decimal_part_str = ""
        separator = ""
    return (
        r"$\mathdefault{"
        + prefix
        + decimal_part_str
        + separator
        + r"{10}^{"
        + (sign if sign == "-" else "")
        + str(exponent)
        + r"}}$"
    )


def fix_label_length(labels: list[str], max_length: int = 20) -> list[str]:
    """
    Ensure that all labels in a list are shorter than the specified length.
    Truncated labels have `...` appended to them.

    :param labels: A list of textual labels.
    :return: A list of possibly truncated labels.
    """
    fixed_labels = []
    for _l in labels:
        if len(_l) <= max_length:
            fixed_labels += [_l]
        else:
            fixed_labels += [_l[: max_length - 3] + "..."]
    return fixed_labels


def update_bars_width(ax: Axes, percentage_width: float = 1) -> None:
    """
    Given an axis with a barplot, scale the width of each bar to the provided percentage,
      and align them to their center.

    :param ax: Axis where bars are located.
    :param percentage_width: Percentage width to which bars are rescaled. By default, do not change their size.
    """
    for patch in [p for p in ax.patches if isinstance(p, Rectangle)]:
        current_width = patch.get_width()
        diff = current_width - percentage_width
        # Change the bar width
        patch.set_width(percentage_width)
        # Recenter the bar
        patch.set_x(patch.get_x() + 0.5 * diff)


def add_labels(
    ax: Axes,
    labels: Optional[list[float]] = None,
    vertical_offsets: Optional[list[float]] = None,
    patch_num: Optional[list[int]] = None,
    fontsize: int = 14,
    rotation: int = 0,
    skip_zero: bool = False,
    format_str: str = "{:.2f}x",
    label_color: str = "#2f2f2f",
    max_only: bool = False,
    skip_bars: int = 0,
    max_bars: Optional[int] = None,
    skip_value: Optional[float] = None,
    skip_threshold: float = 1e-6,
    skip_nan_bars: bool = True,
    max_height: Optional[float] = None,
) -> None:
    """
    Add labels above barplots.

    :param ax: Current axis, it is assumed that each ax.Patch is a bar over which we want to add a label.
    :param labels: Optional labels to add. If not present, add the bar height.
    :param vertical_offsets: Additional vertical offset for each label.
        Useful when displaying error bars (see @get_upper_ci_size), and for fine tuning.
    :param patch_num: Indices of patches to which we add labels, if some of them should be skipped.
    :param fontsize: Size of each label.
    :param rotation: Rotation of the labels (e.g. `90` for 90°).
    :param skip_zero: If True, don't put a label over the first bar.
    :param format_str: Format of each label, by default use speedup (e.g. 2.10x).
    :param label_color: Hexadecimal color used for labels.
    :param max_only: Add only the label with highest value.
    :param skip_bars: Start adding labels after the specified number of bars.
    :param max_bars: Don't add labels after the specified bar.
    :param skip_value: Don't add labels equal to the specified value.
    :param skip_threshold: Threshold used to determine if a label's value is close enough to `skip_value`
        and should be skipped
    :param skip_nan_bars: If True, skip bars with NaN height when placing labels.
    :param max_height: If present, place labels at this maximum specified height (e.g. the y axis limit).
    """
    if not vertical_offsets:
        # 5% above each bar, by default;
        vertical_offsets = [ax.get_ylim()[1] * 0.05] * len(ax.patches)
    if not labels:
        labels = [p.get_height() for p in ax.patches]
        if max_only:
            argmax = np.argmax(labels)
    patches = []
    if not patch_num:
        patches = ax.patches
    else:
        patches = [p for i, p in enumerate(ax.patches) if i in patch_num]
    if skip_nan_bars:
        labels = [_l for _l in labels if not pd.isna(_l)]
        patches = [p for p in patches if not pd.isna(p.get_height())]

    # Iterate through the list of axes' patches
    for i, p in enumerate(patches[skip_bars:max_bars]):
        if (
            labels[i]
            and (i > 0 or not skip_zero)
            and (not max_only or i == argmax)
            and i < len(labels)
            and i < len(vertical_offsets)
        ):
            if skip_value and np.abs(labels[i] - skip_value) < skip_threshold:
                continue  # Skip labels equal to the specified value;
            height = vertical_offsets[i] + p.get_height()
            if max_height is not None and height > max_height:
                height = max_height
            ax.text(
                p.get_x() + p.get_width() / 2,
                height,
                format_str.format(str(labels[i])),
                fontsize=fontsize,
                color=label_color,
                ha="center",
                va="bottom",
                rotation=rotation,
            )


def add_labels_to_bars(
    axes: list[Axes],
    font_size: float = DEFAULT_FONT_SIZE,
    label_format_str: Callable[[Union[float, str]], str] = lambda x: f"{x:.2f}",
) -> list[Axes]:
    # Add the value on top of each bar;
    for ax in axes:
        for p in ax.patches:
            if isinstance(p, Rectangle):
                height = p.get_height()
                if height > 0 and p.get_width() > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.0,
                        height - 0.07,
                        label_format_str(height),
                        ha="center",
                        fontsize=font_size,
                    )
    return axes


def assemble_filenames_to_save_plot(
    directory: Union[str, Path],
    plot_name: str,
    file_format: Union[str, list[str]] = ["pdf", "png"],
    add_timestamp_prefix_to_plot_name: bool = True,
    timestamp_prefix_for_plot_name: str = "%Y-%m-%d_%H-%M-%S_",
    store_plot_into_timestamp_subfolder: bool = True,
    timestamp_format_for_subfolder: str = "%Y-%m-%d",
) -> list[Path]:
    """
    Provide an easy interface to generate paths where plots are stored, using a consistent format
    that allows to easily identify them, and storing them with multiple extensions.
    A plot with name `plot_name` is stored in the `directory` folder.
    Additionally, a timestamp prefix can be added to the plot name, and the plot can be stored in a subfolder.
    The output is a list of paths, one for each extension.
    This function does not create directories, that's responsibility of the caller.

    Examples of generated paths:
    * `directory/plot_name.pdf`, `directory/plot_name.png`
    * `directory/%Y-%m-%d_%H-%M-%S_plot_name.pdf`, `directory/%Y-%m-%d_%H-%M-%S_plot_name.png`
    * `directory/%Y-%m-%d/plot_name.pdf`, `directory/%Y-%m-%d/plot_name.png`
    * `directory/%Y-%m-%d/%Y-%m-%d_%H-%M-%S_plot_name.pdf`, `directory/%Y-%m-%d/%Y-%m-%d_%H-%M-%S_plot_name.png`

    :param directory: Full path to the directory where the folders are stored.
        The parent of this directory must exist, while the directory itself might not exist yet.
    :param plot_name: Name of the plot, without extension.
    :param file_format: List of extensions used to store the plot.
    :param add_timestamp_prefix_to_plot_name: If True, add a timestamp prefix to the plot name.
    :param timestamp_prefix_for_plot_name: Format of the timestamp prefix.
        Used only if `add_timestamp_prefix_to_plot_name` is True.
        The prefix can be a format string representing a timestamp.
        If it's not a valid format string, the prefix is used as-it-is.
    :param store_plot_into_timestamp_subfolder: If True, store the plot in a subfolder with the current date.
    :param timestamp_format_for_subfolder: Format of the timestamp used for the subfolder.
        Used only if `store_plot_into_timestamp_subfolder` is True.
        The prefix can be a format string representing a timestamp.
        If it's not a valid format string, the prefix is used as-it-is.
    :raises ValueError: If the parent directory of `directory` does not exist.
    :return: A list of paths where plots can be stored, one for each extension.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.parent.exists():
        raise ValueError(f"❌ the parent directory {directory.parent} of {directory} does not exist.")
    # Obtain a single timestamp. Try formatting prefix and subfolder if necessary.
    # If formatting fails (e.g. the format string is not valid), use the prefix as-it-is;
    if add_timestamp_prefix_to_plot_name or store_plot_into_timestamp_subfolder:
        timestamp = datetime.today()
        if add_timestamp_prefix_to_plot_name:
            formatted_prefix = timestamp.strftime(timestamp_prefix_for_plot_name)
            plot_name = formatted_prefix + plot_name
        if store_plot_into_timestamp_subfolder:
            formatted_folder = timestamp.strftime(timestamp_format_for_subfolder)
            directory = directory / formatted_folder
    # Assemble the filenames;
    return [directory / f"{plot_name}.{e}" for e in file_format]


def save_plot(
    file_name: Union[str, list[str], Path, list[Path]],
    figure: Optional[Figure] = None,
    dpi: int = 300,
    remove_white_margin: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    :param file_name: One or more absolute file names where the plot is store.
        It is possible to pass multiple paths since one might want to save the same plot with multiple
        extensions, or in multiple locations.
    :param figure: A specific figure to save. If None, save the last plot that has been drawn.
    :param dpi: DPI of the image, when saved as a raster image format such as PNG.
    :param remove_white_margin: If True, remove the white margin around the plot.
        Suitable to plot images without any border around them.
    :param verbose: If True, print information about where the plots have been stored.
    :param kwargs: Other arguments passed to `plt.savefig`.
    """
    if isinstance(file_name, str):
        file_name = [file_name]
    elif isinstance(file_name, Path):
        file_name = [file_name]
    file_name = [Path(_f) for _f in file_name]
    for _f in file_name:
        if not _f.parent.exists():
            _f.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"👉 created directory {_f.parent} to save plots")

    savefig_kwargs: dict[str, Any] = kwargs | {"dpi": dpi}
    if remove_white_margin:
        savefig_kwargs["bbox_inches"] = "tight"
        savefig_kwargs["pad_inches"] = 0

    for _f in file_name:
        if figure is not None:
            figure.savefig(_f, **savefig_kwargs)
        else:  # Save the current plot;
            plt.savefig(_f, **savefig_kwargs)
        if verbose:
            print(f"💡 saved plot to {_f}")
