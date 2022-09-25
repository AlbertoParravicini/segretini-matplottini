#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:14:53 2020

Example of Ridge plot, inspired by https://seaborn.pydata.org/examples/kde_ridgeplot.html

@author: aparravi
"""


from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from segretini_matplottini.plot.ridgeplot import ridgeplot
from segretini_matplottini.utils.data_utils import \
    remove_outliers_from_dataframe_ci
from segretini_matplottini.utils.plot_utils import save_plot

##############################
# Setup ######################
##############################


PLOT_DIR = (Path(__file__).parent.parent / "plots").resolve()
DATA_DIR = (Path(__file__).parent.parent / "data").resolve()


##############################
# Load data and plot #########
##############################


def load_data() -> pd.DataFrame:
    # Load data. The input dataframe is a collection of execution times of different benchmarks.
    # Each benchmark has around 100 samples,
    # and it has been measure before transformations ("exec_time_1_us") and after ("exec_time_2_us");
    data = pd.read_csv(DATA_DIR / "ridgeplot_data.csv")

    # Compute relative execution time before and after transformations and remove outliers.
    # Also assign row/column identifiers to each benchmark for the ridgeplot;
    data = remove_outliers_from_dataframe_ci(data, "exec_time_1_us", groupby="name", debug=True)
    data = remove_outliers_from_dataframe_ci(data, "exec_time_2_us", groupby="name", debug=True)

    # Add relative execution time;
    data["rel_time_1"] = 1
    data["rel_time_2"] = 1
    for _, g in data.groupby("name", as_index=False):
        data.loc[g.index, "rel_time_1"] = g["exec_time_1_us"] / np.mean(g["exec_time_1_us"])
        data.loc[g.index, "rel_time_2"] = g["exec_time_2_us"] / np.mean(g["exec_time_1_us"])
    # Rename columns
    data = data.rename(columns={"rel_time_1": "distribution_1", "rel_time_2": "distribution_2"})
    return data


def plot_1(data: pd.DataFrame) -> sns.FacetGrid:
    return ridgeplot(
        data,
        compact_layout=True,
        xlabel="Relative Execution Time",
        legend_labels=["Before transformations", "After transformations"],
        plot_confidence_intervals=False,
        xlimits=(0.7, 1.3),
    )


def plot_2(data: pd.DataFrame) -> sns.FacetGrid:
    return ridgeplot(
        data,
        compact_layout=False,
        xlabel="Relative Execution Time",
        legend_labels=["Before transformations", "After transformations"],
        xlimits=(0.7, 1.3),
    )


##############################
# Main #######################
##############################

if __name__ == "__main__":

    data = load_data()
    grid = plot_1(data)
    save_plot(PLOT_DIR, "ridgeplot_compact.{}")
    grid = plot_2(data)
    save_plot(PLOT_DIR, "ridgeplot_large.{}")
