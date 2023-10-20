from pathlib import Path

import numpy as np
import pandas as pd

from segretini_matplottini.plot import ridgeplot, ridgeplot_compact
from segretini_matplottini.utils import (
    assemble_filenames_to_save_plot,
    remove_outliers_from_dataframe_ci,
    save_plot,
)

##############################
# Setup ######################
##############################


PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


##############################
# Load data and plot #########
##############################


def load_data() -> pd.DataFrame:
    # Load data. The input dataframe is a collection of execution times of different benchmarks.
    # Each benchmark has around 100 samples,
    # and it has been measure before transformations ("exec_time_1_us") and after ("exec_time_2_us");
    data: pd.DataFrame = pd.read_csv(DATA_DIR / "ridgeplot_data.csv")

    # Compute relative execution time before and after transformations and remove outliers.
    # Also assign row/column identifiers to each benchmark for the ridgeplot;
    data = remove_outliers_from_dataframe_ci(data, "exec_time_1_us", groupby=["name"], verbose=True)
    data = remove_outliers_from_dataframe_ci(data, "exec_time_2_us", groupby=["name"], verbose=True)

    # Add relative execution time;
    data["rel_time_1"] = 1.0
    data["rel_time_2"] = 1.0
    for _, g in data.groupby("name", as_index=False):
        data.loc[g.index, "rel_time_1"] = g["exec_time_1_us"] / np.mean(g["exec_time_1_us"])
        data.loc[g.index, "rel_time_2"] = g["exec_time_2_us"] / np.mean(g["exec_time_1_us"])
    # Rename columns
    data = data.rename(columns={"rel_time_1": "distribution_1", "rel_time_2": "distribution_2"})
    return data


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    ridgeplot(
        data,
        xlabel="Relative execution time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="ridgeplot_large",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
    ridgeplot_compact(
        data,
        xlabel="Relative execution time",
        legend_labels=("Before transformations", "After transformations"),
        xlimits=(0.7, 1.3),
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="ridgeplot_compact",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
