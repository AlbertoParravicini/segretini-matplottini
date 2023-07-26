from pathlib import Path

import pandas as pd
import pytest

from segretini_matplottini.plot import barplot_for_multiple_categories
from segretini_matplottini.utils import (
    add_arrow_to_barplot,
    add_labels_to_bars,
)
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_data.csv")


@save_tmp_plot
def test_default(data: pd.DataFrame) -> None:
    barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value",
        hue="model",
    )


@save_tmp_plot
def test_custom_parameters(data: pd.DataFrame) -> None:
    barplot_for_multiple_categories(
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


@save_tmp_plot
def test_custom_parameters_no_averages(data: pd.DataFrame) -> None:
    barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value",
        hue="model",
        ylimits=(0, 1),
        xlabel="Model",
        ylabel="Value",
        add_bars_for_averages=False,
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


@save_tmp_plot
def test_no_hue(data: pd.DataFrame) -> None:
    barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value",
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


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels(data: pd.DataFrame) -> None:
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
    ax = add_labels_to_bars(axes=[ax], font_size=DEFAULT_FONT_SIZE - 4)[0]
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True)


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels_no_hue(data: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data,
        x="experiment",
        y="value",
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
