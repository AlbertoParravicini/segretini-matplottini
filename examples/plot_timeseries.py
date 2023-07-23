import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from segretini_matplottini.plot import timeseries
from segretini_matplottini.utils import save_plot

##############################
# Setup ######################
##############################

PLOT_DIR = (Path(__file__).parent.parent / "plots").resolve()
DATA_DIR = (Path(__file__).parent.parent / "data").resolve()

##############################
# Load data and plot #########
##############################


def load_data() -> pd.Series:
    data = pd.read_csv(DATA_DIR / "timeseries_data.csv")
    data = data.iloc[:, 0]
    data = data / data.max()
    # Format the index as a timestamp
    frames_per_sec = 24
    timestamps_sec = data.index / frames_per_sec
    data.index = pd.to_datetime(
        pd.Series(timestamps_sec).apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S.%f"))
    ) - datetime.timedelta(hours=1)
    # Make data less dense
    data = data[::60]
    # Compute a moving average
    data = data.rolling(20).mean().dropna()
    return data


def plot(data: np.ndarray) -> tuple[Figure, Axis]:
    return timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        line_width=0.6,
        minutes_interval=2,
        fill=True,
        dark_background=True,
    )


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    fig, ax = plot(data)
    save_plot(PLOT_DIR, "timeseries.{}")
