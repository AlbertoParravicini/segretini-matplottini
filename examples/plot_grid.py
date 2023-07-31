"""
Create four plots with a uniform style and layout, and assemble them into a grid.
This is done to create a grid of plots to show in the README.md file.

Store and reload the four plots as PNGs, put them as a horizontal grid.
If you were to compose plots in a paper, 
you should create a grid in LaTex and load plots as PDFs, to avoid quality loss;
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plot_barplot import load_data_2 as load_data_barplot  # type: ignore
from plot_binary_classification import (  # type: ignore
    generate_data as load_data_binary_classification,
)
from plot_correlation_scatterplot import (  # type: ignore
    load_data as load_data_correlation_scatterplot,
)
from plot_ridgeplot import load_data as load_data_ridgeplot  # type: ignore
from plot_roofline import load_data_2 as load_data_roofline  # type: ignore
from plot_timeseries import load_data_2 as load_data_stem  # type: ignore

from segretini_matplottini.plot import (
    barplot_for_multiple_categories,
    binary_classification,
    correlation_scatterplot,
    ridgeplot,
    roofline,
    timeseries,
)
from segretini_matplottini.utils import (
    add_arrow_to_barplot,
    add_labels_to_bars,
    assemble_filenames_to_save_plot,
    compute_relative_performance,
    get_labels_for_bars,
    save_plot,
)
from segretini_matplottini.utils.colors import (
    GREEN_AND_PINK_TONES,
    PALETTE_ORANGE_BASELINE_AND_GREEN_TONES,
)
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE

#########
# Setup #
#########

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

FIGURE_SIZE = (3.5, 3.5)

############
# Plotting #
############


def plot_correlation_scatterplot(output_dir: Path) -> None:
    # Axes limits used in the plot, change them accordingy to your data;
    xlimits = (-0.2, 0.6)
    ylimits = (-0.1, 0.3)
    data = load_data_correlation_scatterplot()
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        hue="significant",
        xlimits=xlimits,
        ylimits=ylimits,
        density_color=GREEN_AND_PINK_TONES[1],
        regression_color=GREEN_AND_PINK_TONES[0],
        xlabel="Speedup estimate, method A (%)",
        ylabel="Speedup estimate, method B (%)",
        highlight_negative_area=True,
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="correlation_scatterplot",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )
    plt.close()


def plot_roofline(output_dir: Path) -> None:
    # Create a Roofline model, with multiple lines and custom settings;
    data_dict = load_data_roofline()
    roofline(
        data_dict["performance"],
        data_dict["operational_intensity"],
        data_dict["peak_performance"],
        data_dict["peak_bandwidth"],
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Core{'s' if c > 1 else ''}" for c in data_dict["num_cores"]],
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="roofline_stacked",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_stem(output_dir: Path) -> None:
    data = load_data_stem()
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=4,
        minutes_interval_minor_ticks=1,
        dark_background=False,
        draw_style="stem",
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="stem",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_ridgeplot(output_dir: Path) -> None:
    data = load_data_ridgeplot()
    # Keep only one column worth of data
    unique_plots = list(data["name"].unique())
    num_plots = 6
    data = data[data["name"].isin(unique_plots[:num_plots])]
    ridgeplot(
        data,
        xlabel="Relative execution time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
        plot_height=3.5 / num_plots,
        aspect_ratio=num_plots,
        number_of_plot_columns=1,
        bottom_padding=0.2,
        left_padding=0.04,
        right_padding=0.96,
        top_padding=0.99,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="ridgeplot_large",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_binary_classification(output_dir: Path) -> None:
    logits, targets = load_data_binary_classification()
    binary_classification(
        logits,
        targets,
        plot_f1=False,
        plot_roc=False,
        plot_precision_recall=False,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
        figure_size=(3.5, 3.5 * 2 / 3),
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="binary_classification",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_barplot(output_dir: Path) -> None:
    data = load_data_barplot()
    data = compute_relative_performance(
        data, category="model", value="value", baseline_category="model_10", groupby=["experiment"]
    )
    _, ax = barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value_relative_performance",
        hue="model",
        xlabel="",
        ylabel="Relative model performance",
        add_bars_for_averages=True,
        palette=PALETTE_ORANGE_BASELINE_AND_GREEN_TONES,
        hue_category_to_x_tick_label_map={
            "experiment_1": "Experiment 1",
            "experiment_2": "Experiment 2",
            "experiment_3": "Experiment 3",
        },
        x_to_legend_label_map={
            "model_10": "Baseline",
            "model_2": "Model A",
            "model_12": "Model B",
            "model_4": "Model C",
        },
        ylimits=(0, 2),
        y_axis_ticks_count=11,
        figure_size=(3.5, 3.5 * 2 / 3),
        bottom_padding=0.28,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    ax = add_labels_to_bars(
        ax=ax,
        labels=get_labels_for_bars(ax, skip_value=1, label_format_str=lambda x: f"{x:.2f}X"),
        font_size=DEFAULT_FONT_SIZE - 5,
        location="below",
    )
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True, left_margin_to_add=0.1)
    # Update the existing legend to make it more centered;
    ax.get_legend().set_bbox_to_anchor((0.55, 0.09))
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="barplot_for_multiple_categories",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


########
# Main #
########

if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        plot_correlation_scatterplot(output_dir)
        plot_roofline(output_dir)
        plot_stem(output_dir)
        plot_ridgeplot(output_dir)
        # Load the four plots as PNGs, put them as a horizontal grid.
        # This is done to create a grid of plots to show in the README.md file,
        # but if you were to compose plots in a paper,
        # you should create a grid in LaTex and load plots as PDFs, to avoid quality loss;
        fig, axes = plt.subplots(1, 4, figsize=(3.5 * 2, 3.5 * 2), dpi=900, gridspec_kw=dict(hspace=0, wspace=0))
        for i, plot_name in enumerate(["correlation_scatterplot", "roofline_stacked", "stem", "ridgeplot_large"]):
            ax: Axes = axes.flat[i]
            ax.imshow(plt.imread(output_dir / f"{plot_name}.png"))
            ax.axis("off")
        save_plot(
            assemble_filenames_to_save_plot(
                directory=PLOT_DIR,
                plot_name="grid",
                add_timestamp_prefix_to_plot_name=False,
                store_plot_into_timestamp_subfolder=False,
            ),
            verbose=True,
            remove_white_margin=True,
            dpi=900,
        )
    # Another grid, with two other examples;
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        plot_binary_classification(output_dir)
        plot_barplot(output_dir)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(2.7 * 2, 1.8),
            dpi=900,
            gridspec_kw=dict(hspace=0, wspace=0.1, left=0.03, right=0.97, top=1, bottom=0),
        )
        for i, plot_name in enumerate(["binary_classification", "barplot_for_multiple_categories"]):
            ax = axes.flat[i]
            ax.imshow(plt.imread(output_dir / f"{plot_name}.png"))
            ax.axis("off")
        save_plot(
            assemble_filenames_to_save_plot(
                directory=PLOT_DIR,
                plot_name="grid_2",
                add_timestamp_prefix_to_plot_name=False,
                store_plot_into_timestamp_subfolder=False,
            ),
            verbose=True,
            dpi=900,
        )
