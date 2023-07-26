from pathlib import Path

import pandas as pd

from segretini_matplottini.plot import barplot_for_multiple_categories
from segretini_matplottini.utils import (
    add_arrow_to_barplot,
    add_labels_to_bars,
    assemble_filenames_to_save_plot,
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


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_data.csv")


if __name__ == "__main__":
    data = load_data()
    fig, ax = barplot_for_multiple_categories(
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
    ax = add_labels_to_bars(axes=[ax], font_size=DEFAULT_FONT_SIZE - 4)[0]
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="barplot_for_multiple_categories",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
