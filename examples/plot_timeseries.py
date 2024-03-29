import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from segretini_matplottini.plot import timeseries
from segretini_matplottini.utils import assemble_filenames_to_save_plot, save_plot

##############################
# Setup ######################
##############################

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

##############################
# Load data and plot #########
##############################


def load_data_1() -> pd.Series:
    data_csv = pd.read_csv(DATA_DIR / "timeseries_data.csv")
    data = data_csv.iloc[:, 0]
    data = data / data.max()
    # Format the index as a timestamp
    frames_per_sec = 24
    timestamps_sec = data.index / frames_per_sec
    data.index = pd.to_datetime(  # type: ignore
        pd.Series(timestamps_sec).apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S.%f")),
        format="mixed",
    ) - datetime.timedelta(hours=1)
    # Make data less dense
    data = data[::60]
    # Compute a moving average
    data = data.rolling(20).mean().dropna()
    return data


def load_data_2() -> pd.Series:
    """
    Load some sparse data that can be considered as a video frame-wise annotation.
    """
    data_csv = load_data_1()
    data_csv = data_csv.mask(data < data.quantile(0.5), other=0)
    z: pd.Series = pd.Series(np.zeros(len(data_csv)), index=data_csv.index)
    local_maxima = argrelextrema(data_csv.values, np.greater, order=3)[0]
    z.iloc[local_maxima] = data_csv.iloc[local_maxima]
    return z


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data_1()
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=2,
        minutes_interval_minor_ticks=1,
        fill=True,
        dark_background=True,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="timeseries",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
    data = load_data_2()
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=2,
        minutes_interval_minor_ticks=1,
        dark_background=True,
        draw_style="stem",
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="stem",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
