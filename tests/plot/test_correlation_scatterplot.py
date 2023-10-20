from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from segretini_matplottini.plot import correlation_scatterplot
from segretini_matplottini.utils.colors import TWO_TEAL_TONES

from .utils import save_tmp_plot  # noqa: F401

# Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (-0.2, 0.6)
Y_LIMITS = (-0.1, 0.3)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def data() -> pd.DataFrame:
    # Load data;
    data = pd.read_csv(DATA_DIR / "correlation_scatterplot_data.csv")
    # Remove outliers present in the dataset;
    data = data[data["estimate0"] < 1]
    data = data[data["estimate1"] < 1]
    return data


@save_tmp_plot
def test_default_settings(data: pd.DataFrame) -> None:
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
    )


@save_tmp_plot
def test_custom_settings(data: pd.DataFrame) -> None:
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        hue="significant",
        xlimits=X_LIMITS,
        ylimits=Y_LIMITS,
        scatterplot_palette=TWO_TEAL_TONES,
        xlabel="Speedup estimate, method A (%)",
        ylabel="Speedup estimate, method B (%)",
        highlight_negative_area=True,
    )


@save_tmp_plot
def test_no_regression(data: pd.DataFrame) -> None:
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        plot_regression=False,
    )


@save_tmp_plot
def test_no_kde(data: pd.DataFrame) -> None:
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        plot_kde=False,
    )


@save_tmp_plot
def test_existing_axis(data: pd.DataFrame) -> None:
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        ax=ax,
    )
