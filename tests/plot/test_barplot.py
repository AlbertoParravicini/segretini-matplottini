from pathlib import Path

import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from segretini_matplottini.plot import (
    barplot,
    barplot_for_multiple_categories,
    barplots,
)
from segretini_matplottini.utils import (
    add_arrow_to_barplot,
    add_labels_to_bars,
    get_labels_for_bars,
)
from segretini_matplottini.utils.constants import DEFAULT_FONT_SIZE

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def data_1() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_data.csv")


@pytest.fixture
def data_2() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "barplot_for_multiple_categories_data.csv")


@save_tmp_plot
def test_default(data_1: pd.DataFrame) -> None:
    fig, ax = barplot(
        data_1,
        x="model",
        y="value",
    )
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 3


@save_tmp_plot
def test_custom_parameters(data_1: pd.DataFrame) -> None:
    _, ax = barplot(
        data_1,
        x="model",
        y="value",
        ylimits=(0, 1),
        xlabel="Model",
        ylabel="Value",
        add_legend=True,
        x_to_legend_label_map={
            "model_10": "A",
            "model_2": "B",
            "model_12": "C",
            "model_4": "D",
        },
    )
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 3


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels(data_1: pd.DataFrame) -> None:
    _, ax = barplot(
        data_1,
        x="model",
        y="value",
        ylimits=(0, 1),
        xlabel="Model",
        ylabel="Value",
        add_legend=True,
        x_to_legend_label_map={
            "model_10": "A",
            "model_2": "B",
            "model_12": "C",
            "model_4": "D",
        },
    )
    ax = add_labels_to_bars(ax=ax, labels=get_labels_for_bars(ax), font_size=DEFAULT_FONT_SIZE - 4)
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True)
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 3


@save_tmp_plot
def test_default_barplots(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
    )
    assert len(axes) == 2
    assert len(axes[0]) == 3
    assert len(axes[1]) == 2
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_default_barplots_one_row(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        number_of_rows=1,
    )
    assert len(axes) == 1
    assert len(axes[0]) == 5
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_default_barplots_one_col(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        number_of_columns=1,
    )
    assert len(axes) == 5
    assert all(len(ax) == 1 for ax in axes)
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_default_barplots_two_col(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        number_of_columns=2,
    )
    assert len(axes) == 3
    assert len(axes[0]) == 2
    assert len(axes[1]) == 2
    assert len(axes[2]) == 1
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_custom_parameters_barplots(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        ylimits=(0, 1),
        x_to_legend_label_map={"A": "Model A", "B": "Model B", "C": "Model C"},
        category_to_y_label_map={
            "metric_10": "Metric 1",
            "metric_2": "Metric 2",
            "metric_4": "Metric 3",
            "metric_1": "Metric 4",
            "metric_3": "Metric 5",
        },
    )
    assert len(axes) == 2
    assert len(axes[0]) == 3
    assert len(axes[1]) == 2
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_single_plot_barplots(data_1: pd.DataFrame) -> None:
    data_1 = data_1.groupby(["model"]).mean(numeric_only=True).reset_index()
    data_1["metric"] = "metric_1"
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        ylimits=(0, 1),
        x_to_legend_label_map={"A": "Model A", "B": "Model B", "C": "Model C"},
        category_to_y_label_map={
            "metric_1": "Metric 4",
        },
    )
    assert len(axes) == 1
    assert len(axes[0]) == 1
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels_barplots(data_1: pd.DataFrame) -> None:
    _, axes = barplots(
        data_1,
        x="model",
        y="value",
        category="metric",
        x_to_legend_label_map={"A": "Model A", "B": "Model B", "C": "Model C"},
        category_to_y_label_map={
            "metric_10": "Metric 1",
            "metric_2": "Metric 2",
            "metric_4": "Metric 3",
            "metric_1": "Metric 4",
            "metric_3": "Metric 5",
        },
    )
    for ax_i in axes:
        for ax_j in ax_i:
            ax_j = add_labels_to_bars(
                ax=ax_j, labels=get_labels_for_bars(ax_j), font_size=DEFAULT_FONT_SIZE - 4, location="below"
            )
            ax_j = add_arrow_to_barplot(ax=ax_j, higher_is_better=True, left_margin_to_add=0.4)
    assert len(axes) == 2
    assert len(axes[0]) == 3
    assert len(axes[1]) == 2
    for ax_i in axes:
        for ax_j in ax_i:
            rectangles = [p for p in ax_j.patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 3


@save_tmp_plot
def test_default_for_multiple_categories(data_2: pd.DataFrame) -> None:
    barplot_for_multiple_categories(
        data_2,
        x="experiment",
        y="value",
        hue="model",
    )


@save_tmp_plot
def test_custom_parameters_for_multiple_categories(data_2: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data_2,
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
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 16


@save_tmp_plot
def test_custom_parameters_no_averages_for_multiple_categories(data_2: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data_2,
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
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 12


@save_tmp_plot
def test_no_hue_for_multiple_categories(data_2: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data_2,
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
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 4


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels_for_multiple_categories(data_2: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data_2,
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
    ax = add_labels_to_bars(ax=ax, labels=get_labels_for_bars(ax), font_size=DEFAULT_FONT_SIZE - 4)
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True)
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 16


@save_tmp_plot
def test_custom_parameters_and_arrow_and_labels_no_hue_for_multiple_categories(data_2: pd.DataFrame) -> None:
    _, ax = barplot_for_multiple_categories(
        data_2,
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
    ax = add_labels_to_bars(ax=ax, labels=get_labels_for_bars(ax), font_size=DEFAULT_FONT_SIZE - 4)
    ax = add_arrow_to_barplot(ax=ax, higher_is_better=True)
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 4
