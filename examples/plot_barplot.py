from pathlib import Path

import pandas as pd

from segretini_matplottini.plot import (
    barplot,
    barplot_for_multiple_categories,
    barplots,
)
from segretini_matplottini.utils import (
    add_arrow_to_barplot,
    add_labels_to_bars,
    assemble_filenames_to_save_plot,
    get_labels_for_bars,
    save_plot,
)
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE

##############################
# Setup ######################
##############################

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

##############################
# Load data and plot #########
##############################


def load_data_1() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_data.csv")


def load_data_2() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_for_multiple_categories_data.csv")


if __name__ == "__main__":
    # Plot a single barplot
    data = load_data_1()
    _, ax = barplot(data, x="model", y="value", add_legend=True, ylabel="Metric A", xlabel="Model")
    ax = add_labels_to_bars(ax=ax, labels=get_labels_for_bars(ax), font_size=DEFAULT_FONT_SIZE - 4, location="below")
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True, left_margin_to_add=0.2)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="barplot",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
    # Plot a grid of barplots
    _, axes = barplots(
        data,
        x="model",
        y="value",
        category="metric",
        add_bars_for_averages=True,
        x_to_legend_label_map={"A": "Model A", "B": "Model B", "C": "Model C"},
        category_to_y_label_map={
            "metric_10": "Metric 1",
            "metric_2": "Metric 2",
            "metric_4": "Metric 3",
            "metric_1": "Metric 4",
            "metric_3": "Metric 5",
        },
    )
    # Add arrows and labels. For the labels, we display the relative performance instead of the absolute value.
    # We skip the baseline labels, since their relative performance is always 1X.
    for ax_i in axes:
        for ax_j in ax_i:
            ax_j = add_labels_to_bars(
                ax=ax_j,
                labels=get_labels_for_bars(
                    ax_j, normalize_wrt_minimum=True, label_format_str=lambda x: f"{x:.2f}X", skip_value=1
                ),
                font_size=DEFAULT_FONT_SIZE - 5,
                location="below",
            )
            ax_j = add_arrow_to_barplot(ax=ax_j, higher_is_better=True, left_margin_to_add=0.4)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="barplots",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
    # Plot multiple barplots on the same row, grouped by different categories
    data = load_data_2()
    _, ax = barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value",
        hue="model",
        ylimits=(0, 1),
        xlabel="Model",
        ylabel="Value",
        add_bars_for_averages=True,
        hue_category_to_x_tick_label_map={
            "experiment_1": "Experiment 1",
            "experiment_2": "Experiment 2",
            "experiment_3": "Experiment 3",
        },
        x_to_legend_label_map={
            "model_10": "A",
            "model_2": "B",
            "model_12": "C",
            "model_4": "D",
        },
    )
    ax = add_labels_to_bars(ax=ax, labels=get_labels_for_bars(ax), font_size=DEFAULT_FONT_SIZE - 4, location="below")
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True, left_margin_to_add=0.1)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="barplot_for_multiple_categories",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
