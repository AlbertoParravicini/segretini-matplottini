from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from segretini_matplottini.plot.correlation_scatterplot import \
    correlation_scatterplot
from segretini_matplottini.utils.colors import PALETTE_G
from segretini_matplottini.utils.plot_utils import save_plot

##############################
# Setup ######################
##############################

# Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (-0.2, 0.6)
Y_LIMITS = (-0.1, 0.3)

# Color palette used for plotting;
PALETTE = [PALETTE_G[0], PALETTE_G[2]]

PLOT_DIR = (Path(__file__).parent.parent / "plots").resolve()
DATA_DIR = (Path(__file__).parent.parent / "data").resolve()

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
        hue="significant",
        xlimits=X_LIMITS,
        ylimits=Y_LIMITS,
        palette=PALETTE,
        xlabel="Speedup estimate, method A (%)",
        ylabel="Speedup estimate, method B (%)",
    )


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    fig, ax = plot(data)
    save_plot(PLOT_DIR, "correlation_scatterplot.{}")
