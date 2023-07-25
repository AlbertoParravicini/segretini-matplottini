import datetime
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from segretini_matplottini.plot import timeseries
from segretini_matplottini.utils import save_plot

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(autouse=True)
def reset_plot_style() -> None:
    # Reset plotting settings
    plt.rcdefaults()
    return


def save_tmp_plot() -> None:
    plot_dir = Path(__file__).parent.parent.parent / "plots" / "tests"
    plot_dir.mkdir(parents=True, exist_ok=True)
    caller_name = inspect.stack()[1][3]
    save_plot(plot_dir / f"{Path(__file__).stem}_{caller_name}.png")


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


def test_default(data: pd.Series) -> None:
    timeseries(
        data,
    )
    save_tmp_plot()


def test_custom_settings(data: pd.Series) -> None:
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval=2,
        fill=True,
        dark_background=True,
    )
    save_tmp_plot()


def test_existing_axis(data: pd.Series) -> None:
    _, ax = plt.subplots(1, 1, figsize=(6, 3))
    timeseries(
        data,
        ax=ax,
    )
    save_tmp_plot()
