from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from segretini_matplottini.plot import correlation_scatterplot
from segretini_matplottini.utils import assemble_filenames_to_save_plot, save_plot
from segretini_matplottini.utils.colors import GREEN_AND_PINK_TONES

##############################
# Setup ######################
##############################

# Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (-0.2, 0.6)
Y_LIMITS = (-0.1, 0.3)

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

##############################
# Load data and plot #########
##############################


def load_data() -> pd.DataFrame:
    # Load data;
    data = pd.read_csv(DATA_DIR / "correlation_scatterplot_data.csv")
    # Remove outliers present in the dataset;
    data = data[data["estimate0"] < 1]
    data = data[data["estimate1"] < 1]
    return data


def plot(data: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    return correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        hue="significant",
        xlimits=X_LIMITS,
        ylimits=Y_LIMITS,
        density_color=GREEN_AND_PINK_TONES[1],
        regression_color=GREEN_AND_PINK_TONES[0],
        xlabel="Speedup estimate, method A (%)",
        ylabel="Speedup estimate, method B (%)",
        highlight_negative_area=True,
    )


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    fig, ax = plot(data)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="correlation_scatterplot",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
