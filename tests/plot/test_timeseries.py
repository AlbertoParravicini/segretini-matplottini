import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from segretini_matplottini.plot import timeseries

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def data() -> pd.Series:
    data = pd.read_csv(DATA_DIR / "timeseries_data.csv")
    data = data.iloc[:, 0]
    data = data / data.max()
    # Format the index as a timestamp
    frames_per_sec = 24
    timestamps_sec = data.index / frames_per_sec
    data.index = pd.to_datetime(
        pd.Series(timestamps_sec).apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S.%f")),
        format="mixed",
    ) - datetime.timedelta(hours=1)
    # Make data less dense
    data = data[::60]
    # Compute a moving average
    data = data.rolling(20).mean().dropna()
    return data


@save_tmp_plot
def test_default(data: pd.Series) -> None:
    timeseries(
        data,
    )


@save_tmp_plot
def test_custom_settings(data: pd.Series) -> None:
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=2,
        fill=True,
        dark_background=True,
    )


@save_tmp_plot
def test_existing_axis(data: pd.Series) -> None:
    _, ax = plt.subplots(1, 1, figsize=(6, 3))
    timeseries(
        data,
        ax=ax,
    )


@save_tmp_plot
def test_stem_default(data: pd.Series) -> None:
    timeseries(
        data,
        draw_style="stem",
    )


@save_tmp_plot
def test_stem_custom_settings(data: pd.Series) -> None:
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=2,
        seconds_interval_minor_ticks=30,
        dark_background=True,
        draw_style="stem",
    )
