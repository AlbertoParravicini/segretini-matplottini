from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from segretini_matplottini.utils.colors import BACKGROUND_BLACK
from segretini_matplottini.utils.constants import DEFAULT_DPI, DEFAULT_FONT_SIZE


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
    :param border_width: Line width of the axis borders, and also of the ticks.
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
    # Set the width of the axes borders, and also the width of the ticks,
    # so they have the same style;
    plt.rcParams["axes.linewidth"] = border_width
    plt.rcParams["xtick.major.width"] = border_width
    plt.rcParams["ytick.major.width"] = border_width
    if title_size is not None:
        plt.rcParams["axes.titlesize"] = title_size
    if title_pad is not None:
        plt.rcParams["axes.titlepad"] = title_pad
    if label_size is not None:
        plt.rcParams["axes.labelsize"] = label_size
    # Background color
    if dark_background:
        activate_dark_background()


def adjust_rows_and_columns_to_number_of_plots(
    number_of_plots: int,
    number_of_rows: Optional[int] = None,
    number_of_columns: Optional[int] = None,
) -> tuple[int, int]:
    """
    Adjust the input number of rows and number of columns to match the desired number of plots.
    If either the number of rows or columns is present, the other value is inferred from the number of categories.
    If both values are missing, the number of rows and columns is approximately the square root of the number of plots,
    to provide a grid that is as square as possible.
    If both the number of rows and the number of columns is present, the number of columns is given priority.

    :param number_of_plots: The number of plots to draw.
    :param number_of_rows: Number of rows of the grid of plots. If None, infer it from the number of categories.
    :param number_of_columns: Number of columns of the grid of plots. If None, infer it from the number of categories.
    :return: The adjusted number of rows and columns to use to draw the plots.
    """
    # Obtain the number of rows and columns to plot;
    if number_of_rows is not None and number_of_columns is not None:
        # If both the number of rows and the number of columns is present,
        # give priority to the number of columns;
        _number_of_columns = number_of_columns
        _number_of_rows = int(np.ceil(number_of_plots / _number_of_columns))
        if _number_of_rows != number_of_rows:
            print(
                f"⚠️ both {number_of_rows=} and {number_of_columns=} are specified; "
                f"overriding {number_of_rows=} to {_number_of_rows=}"
            )
    elif number_of_columns is None and number_of_rows is not None:
        _number_of_rows = number_of_rows
        _number_of_columns = int(np.ceil(number_of_plots / _number_of_rows))
    elif number_of_rows is None and number_of_columns is not None:
        _number_of_columns = number_of_columns
        _number_of_rows = int(np.ceil(number_of_plots / _number_of_columns))
    else:
        # The number of rows and columns is approximately the square root of the number of plots;
        _number_of_columns = int(np.ceil(np.sqrt(number_of_plots)))
        _number_of_rows = int(np.ceil(number_of_plots / _number_of_columns))
    return _number_of_rows, _number_of_columns


def add_arrow_to_barplot(
    ax: Axes,
    higher_is_better: bool = True,
    line_width: float = 0.5,
    left_margin_to_add: float = 0.3,
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
    y_start: float = 0.01
    y_end: float = 0.99
    ax.annotate(
        "",
        xy=(x_coord, y_end),
        xytext=(x_coord, y_start),
        arrowprops=dict(
            arrowstyle="->" if higher_is_better else "<-",
            color=arrow_color,
            linewidth=line_width,
        ),
        xycoords=ax.get_xaxis_transform(),
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


def get_labels_for_bars(
    ax: Axes,
    skip_zeros: bool = True,
    skip_nan: bool = True,
    skip_value: Optional[float] = None,
    skip_threshold: float = 1e-6,
    max_only: bool = False,
    label_format_str: Callable[[Union[float, str]], str] = lambda x: f"{x:.2f}",
    normalize_wrt_minimum: bool = False,
) -> list[str]:
    """
    Given a barplot (or a plot containing bars), obtain a list of labels representing the value of each bar.

    :param ax: Axis containing the barplot. Patches that are not rectangles are ignored.
    :param skip_zeros: If True, skip bars with height equal to zero.
    :param skip_nan: If True, skip bars with NaN height.
    :param skip_value: If not None, skip bars with height equal to this value.
    :param skip_threshold: Threshold used to determine if a label's value
        is close enough to `skip_value` and `skip_zero`.
    :param max_only: If True, return only the label of the bar with highest value.
    :param label_format_str: Format of each label, by default use two decimal digits (e.g. 2.10).
    :param normalize_wrt_minimum: If True, normalize the label numeric values w.r.t. the label with smallest value,
        to obtain a relative performance.
    :return: A list of labels, one for each bar. Bars that have been skipped have an empty string as label.
    """
    assert hasattr(ax, "containers"), f"❌ the axis {ax} does not have any container, are you sure it's a barplot?"
    containers: list[BarContainer] = ax.containers
    labels_for_each_category: list[list[str]] = []

    # If plotting bars grouped by a category, they will appear in separate BarContainers.
    # The problem is that the bars of a category are not in the same BarContainer,
    # but bars of the i-th category are in the i-th position of each BarContainer;
    heights_for_each_category: list[list[float]] = []
    # Do not transpose if only one container is present, since all bars are already within that single container;
    transposed_bars: np.ndarray = np.array(containers).T if len(containers) > 1 else np.array(containers)
    for bar_container in transposed_bars:
        rectangles = [p for p in bar_container if isinstance(p, Rectangle)]
        heights = [p.get_height() for p in rectangles]
        if normalize_wrt_minimum:
            min_height = min([h for h in heights if h != 0 and not pd.isna(h)])
            heights = [h / min_height for h in heights]
        heights_for_each_category += [heights]
    # Obtain the labels for each category;
    for heights in heights_for_each_category:
        _labels_for_each_category: list[str] = []
        if max_only:
            # Simpy obtain the height of the highest bar, and set everything else to empty string;
            argmax = np.argmax(heights)
            _labels_for_each_category = ["" for _ in heights]
            _labels_for_each_category[argmax] = label_format_str(heights[argmax])
        else:
            # Get the height of each bar, and format it as a label.
            # Skip it if necessary, adding an empty label;
            for height in heights:
                if skip_zeros and np.abs(height) < skip_threshold:
                    _labels_for_each_category.append("")
                elif skip_nan and pd.isna(height):
                    _labels_for_each_category.append("")
                elif skip_value is not None and np.abs(height - skip_value) < skip_threshold:
                    _labels_for_each_category.append("")
                else:
                    _labels_for_each_category.append(label_format_str(height))
        labels_for_each_category += [_labels_for_each_category]
    # Transpose the labels for each category, so the i-th label correspond to the i-th bar in the original plot;
    return [str(x) for x in np.array(labels_for_each_category).T.reshape(-1).tolist()]


def add_labels_to_bars(
    ax: Axes,
    labels: list[str],
    font_size: float = DEFAULT_FONT_SIZE,
    rotation: float = 0,
    label_color: str = "#2f2f2f",
    location: Literal["above", "below"] = "above",
    vertical_offset_points: float = 0.5,
    do_not_exceed_ylim: bool = True,
    tolerance_for_ylim: float = 0.05,
) -> Axes:
    """
    Add labels to the top of each bar in a barplot.

    :param ax: Axis containing the barplot. Patches that are not rectangles are ignored.
    :param labels: List of labels to add. The number of labels must match the number of bars.
    :param font_size: Font size of the labels
    :param rotation: Rotation of the labels (e.g. `90` for 90°).
    :param label_color: Hexadecimal color used for labels.
    :param location: If "above", add labels above the top of each bar.
        If "below", add labels below the top of each bar.
    :param vertical_offset_points: Vertical padding, as offset points w.r.t. the top of each bar.
    :param do_not_exceed_ylim: If True, labels that would exceed the y-axis limits are added at the limit.
    :param tolerance_for_ylim: Tolerance used to determine if a label's value is close enough to the y-axis limit,
        and should be added at the limit. The tolerance is a percentage of the vertical size of the plot.
        The tolerance is used since values close to the top/bottom would overlap with the border of the plot.
    :return: The axis containing the barplot, with the labels added.
    """
    # Keep only rectangles;
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    # The number of labels and rectangles must match;
    assert len(rectangles) == len(
        labels
    ), f"❌ the number of labels ({len(labels)}) and rectangles ({len(rectangles)}) must match."
    # Compute the vertical padding using the vertical size of the plot;
    _vertical_offset_points = vertical_offset_points
    if location == "below":
        # Invert the padding, and add a bit of extra space to avoid overlapping with the bar;
        _vertical_offset_points = -_vertical_offset_points - 1
    # Get the tolerance for the y-axis limits.
    # We need a tolerance since values close to the top/bottom would overlap with the border of the plot;
    _tolerance_for_ylim = tolerance_for_ylim * (ax.get_ylim()[1] - ax.get_ylim()[0])
    # Add labels
    for label, bar in zip(labels, rectangles):
        height = bar.get_height()
        if do_not_exceed_ylim:
            y_min, y_max = ax.get_ylim()
            # Handle cases where bars exceed (or almost exceed) the y-axis limits;
            if height > y_max:
                height = y_max
            # Only if the location is above, otherwise a label that's fully contained
            # in a bar almost at the top would be moved up and get clipped;
            elif height + _tolerance_for_ylim > y_max and location == "above":
                height = y_max
            # Only if the location is below, since if it's above it does not get clipped;
            elif height - _tolerance_for_ylim < y_min and location == "below":
                height = y_min
        label_x_coordinate = bar.get_x() + bar.get_width() / 2.0
        label_y_coordinate = height
        ax.annotate(
            text=label,
            xy=(label_x_coordinate, label_y_coordinate),  # Coordinates of the label, in data-coordinates;
            xytext=(0, _vertical_offset_points),  # Coordinates of the text, as offset points w.r.t. `xy`;
            textcoords="offset points",
            fontsize=font_size,
            color=label_color,
            rotation=rotation,
            va="bottom" if location == "above" else "top",
            clip_on=False,
            ha="center",
        )
    return ax


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
    :param file_format: One or more extensions used to store the plot.
        Extensions supported by Matplotlib: `eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp`
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
    if isinstance(file_format, str):
        file_format = [file_format]
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
    dpi: int = DEFAULT_DPI,
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
