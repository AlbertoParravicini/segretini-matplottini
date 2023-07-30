from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segretini_matplottini.plot import ridgeplot, ridgeplot_compact
from segretini_matplottini.utils import (
    remove_outliers_from_dataframe_ci,
)

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def data() -> pd.DataFrame:
    # Load data. The input dataframe is a collection of execution times of different benchmarks.
    # Each benchmark has around 100 samples,
    # and it has been measure before transformations ("exec_time_1_us") and after ("exec_time_2_us");
    data = pd.read_csv(DATA_DIR / "ridgeplot_data.csv")

    # Compute relative execution time before and after transformations and remove outliers.
    # Also assign row/column identifiers to each benchmark for the ridgeplot;
    data = remove_outliers_from_dataframe_ci(data, "exec_time_1_us", groupby=["name"], verbose=True)
    data = remove_outliers_from_dataframe_ci(data, "exec_time_2_us", groupby=["name"], verbose=True)

    # Add relative execution time;
    data["rel_time_1"] = 1
    data["rel_time_2"] = 1
    for _, g in data.groupby("name", as_index=False):
        data.loc[g.index, "rel_time_1"] = g["exec_time_1_us"] / np.mean(g["exec_time_1_us"])
        data.loc[g.index, "rel_time_2"] = g["exec_time_2_us"] / np.mean(g["exec_time_1_us"])
    # Rename columns
    data = data.rename(columns={"rel_time_1": "distribution_1", "rel_time_2": "distribution_2"})
    return data


@save_tmp_plot
def test_ridgeplot_default(data: pd.DataFrame) -> None:
    ridgeplot(
        data,
    )


@save_tmp_plot
def test_ridgeplot(data: pd.DataFrame) -> None:
    ridgeplot(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
    )


@save_tmp_plot
def test_ridgeplot_compact_default(data: pd.DataFrame) -> None:
    ridgeplot_compact(
        data,
    )


@save_tmp_plot
def test_ridgeplot_compact(data: pd.DataFrame) -> None:
    ridgeplot_compact(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        xlimits=(0.7, 1.3),
    )


@save_tmp_plot
def test_ridgeplot_one_col(data: pd.DataFrame) -> None:
    ridgeplot(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
        number_of_plot_columns=1,
    )


@save_tmp_plot
def test_ridgeplot_compact_one_col(data: pd.DataFrame) -> None:
    ridgeplot_compact(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        xlimits=(0.7, 1.3),
        number_of_plot_columns=1,
    )


@save_tmp_plot
def test_ridgeplot_three_col(data: pd.DataFrame) -> None:
    ridgeplot(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
        number_of_plot_columns=3,
    )


@save_tmp_plot
def test_ridgeplot_compact_three_col(data: pd.DataFrame) -> None:
    ridgeplot_compact(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        xlimits=(0.7, 1.3),
        number_of_plot_columns=3,
    )
