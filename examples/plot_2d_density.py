from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from segretini_matplottini.utils import save_plot
from segretini_matplottini.utils.colors import PALETTE_G

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
    data = pd.read_csv(DATA_DIR / "density_data.csv")
    return data


def plot(data: pd.DataFrame) -> plt.Axes:
    return sns.scatterplot(
        data=data,
        x="threshold_1",
        y="threshold_2",
        hue="p",
        size="p",
        # hue="significant",
        # xlimits=X_LIMITS,
        # ylimits=Y_LIMITS,
        # palette=PALETTE,
        # xlabel="Speedup estimate, method A (%)",
        # ylabel="Speedup estimate, method B (%)",
    )


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    ax = plot(data)
    save_plot(PLOT_DIR, "2d_density.{}")
