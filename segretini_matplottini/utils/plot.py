import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from segretini_matplottini.utils.colors import BACKGROUND_BLACK


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
    if dark_background:
        plt.style.use("dark_background")
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
        plt.rcParams["axes.facecolor"] = BACKGROUND_BLACK
        plt.rcParams["savefig.facecolor"] = BACKGROUND_BLACK


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


def update_bars_width(ax: Axis, percentage_width: float = 1) -> None:
    """
    Given an axis with a barplot, scale the width of each bar to the provided percentage,
      and align them to their center.

    :param ax: Axis where bars are located.
    :param percentage_width: Percentage width to which bars are rescaled. By default, do not change their size.
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - percentage_width
        # Change the bar width
        patch.set_width(percentage_width)
        # Recenter the bar
        patch.set_x(patch.get_x() + 0.5 * diff)


def add_labels(
    ax: Axis,
    labels: Optional[list[str]] = None,
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
    ...
    # if not vertical_offsets:
    #     # 5% above each bar, by default;
    #     vertical_offsets = [ax.get_ylim()[1] * 0.05] * len(ax.patches)
    # if not labels:
    #     labels = [p.get_height() for p in ax.patches]
    #     if max_only:
    #         argmax = np.argmax(labels)
    # patches = []
    # if not patch_num:
    #     patches = ax.patches
    # else:
    #     patches = [p for i, p in enumerate(ax.patches) if i in patch_num]
    # if skip_nan_bars:
    #     labels = [l for l in labels if not pd.isna(l)]
    #     patches = [p for p in patches if not pd.isna(p.get_height())]

    # # Iterate through the list of axes' patches
    # for i, p in enumerate(patches[skip_bars:max_bars]):
    #     if (
    #         labels[i]
    #         and (i > 0 or not skip_zero)
    #         and (not max_only or i == argmax)
    #         and i < len(labels)
    #         and i < len(vertical_offsets)
    #     ):
    #         if skip_value and np.abs(labels[i] - skip_value) < skip_threshold:
    #             continue  # Skip labels equal to the specified value;
    #         height = vertical_offsets[i] + p.get_height()
    #         if max_height is not None and height > max_height:
    #             height = max_height
    #         ax.text(
    #             p.get_x() + p.get_width() / 2,
    #             height,
    #             format_str.format(labels[i]),
    #             fontsize=fontsize,
    #             color=label_color,
    #             ha="center",
    #             va="bottom",
    #             rotation=rotation,
    #


def assemble_output_directory_name(
    directory: Union[str, Path], date: Optional[str] = None, create_date_dir: bool = True
) -> str:
    """
    Create the output directory where plots are saved.
    Optionally, create a subfolder with todays' date formatted as `%Y-%m-%d`,
    and save plots inside it. Parent folders of `directory` are assumed to exist.

    :param directory: Path to the directory where plots are created.
    :param date: If present, create a subfolder with this date inside `directory`.
    :param create_date_dir: If True, and `date` is None, create the date subfolder as `%Y-%m-%d`.
    :return: Path of the folder that has been created, as a string.
    """
    p = Path(directory)
    p.mkdir(parents=False, exist_ok=True)
    if create_date_dir:
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        p = Path(directory) / date
    p.mkdir(parents=False, exist_ok=True)
    return str(p)


def save_plot(
    directory: Union[str, Path],
    filename: str,
    figure: Optional[Figure] = None,
    date: Optional[str] = None,
    create_date_dir: bool = True,
    extension: list = ["pdf", "png"],
    dpi: int = 300,
    remove_white_margin: bool = False,
    **kwargs: Any,
) -> None:
    """
    :param directory: Where the plot is stored.
    :param filename: Name of the plot. It should be of format 'myplot_{}.{}',
        where the first placeholder is used for the date and the second for the extension,
        or 'myplot.{}', or 'myplot.extension'.
    :param figure: A specific figure to save. If None, save the last plot that has been drawn.
    :param date: Date that should appear in the plot filename.
    :param create_date_dir: If True, create a sub-folder with the date. If `date` is None, use `%Y-%m-%d`.
    :param extension: List of extension used to store the plot.
    :param dpi: DPI of the image, when saved as a raster image format such as PNG.
    :param remove_white_margin: If True, remove the white margin around the plot.
        Suitable to plot images without any border around them.
    :param kwargs: Other arguments passed to `plt.savefig`.
    """
    output_folder = assemble_output_directory_name(directory, date, create_date_dir)

    savefig_kwargs: dict[str, Any] = kwargs | {"dpi": dpi}
    if remove_white_margin:
        savefig_kwargs["bbox_inches"] = "tight"
        savefig_kwargs["pad_inches"] = 0

    for e in extension:
        # Format the filename
        try:
            output_filename = filename.format(e)
        except (ValueError, IndexError):
            output_filename = filename.format(date, e)
        if figure is not None:
            figure.savefig(os.path.join(output_folder, output_filename), **savefig_kwargs)
        else:  # Save the current plot;
            plt.savefig(os.path.join(output_folder, output_filename), **savefig_kwargs)
