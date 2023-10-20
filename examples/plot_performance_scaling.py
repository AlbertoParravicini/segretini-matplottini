from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.dates import MonthLocator, YearLocator, num2date
from matplotlib.ticker import FuncFormatter
from sklearn import linear_model

from segretini_matplottini.utils import (
    assemble_filenames_to_save_plot,
    reset_plot_style,
    save_plot,
)

##############################
# Setup ######################
##############################

# Color palette used for plotting;
PALETTE = ["#F0B390", "#90D6B4", "#7BB7BA"]
MARKERS = ["o", "D", "X"]

# Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (datetime(year=1996, month=1, day=1), datetime(year=2022, month=1, day=1))
Y_LIMITS = (0.1, 5 * 1e6)

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

##############################
# Plotting functions #########
##############################


def performance_scaling(
    data: pd.DataFrame, set_axes_limits: bool = True, plot_regression: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """
    :param data: pd.DataFrame with 6 columns:
        "year",
        "performance",
        "kind" ∈ ["compute", "memory", "interconnect"],
        "name" (label shown in the plot, it can be empty),
        "base" (base value used for speedup, it can be empty),
        "comment" (e.g. data source or non-used label, it can be empty).
    :param set_axes_limits: If True, set the axes limits as specified in `X_LIMITS` and `Y_LIMITS`.
    :param plot_regression: If True, plot a linear regression of the values.
    :return: Matplotlib figure and axis containing the plot
    """

    ##############
    # Plot setup #
    ##############

    reset_plot_style()

    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(6, 2.5))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.12, right=0.99)
    ax = fig.add_subplot(gs[0, 0])

    # Set axes limits;
    if set_axes_limits:
        ax.set_xlim(X_LIMITS)  # type: ignore
        ax.set_ylim(Y_LIMITS)

    #################
    # Main plot #####
    #################

    # Measure performance increase over 20 and 2 years;
    kind_increase = {}

    # Add a scatterplot for individual elements of the dataset, and change color based on hardware type;
    ax = sns.scatterplot(
        x="year",
        y="performance",
        hue="kind",
        style="kind",
        palette=PALETTE,
        markers=MARKERS,
        s=15,
        data=data,
        ax=ax,
        edgecolor="#2f2f2f",
        linewidth=0.5,
        zorder=4,
        legend=False,
    )

    # Add a regression plot to highlight the correlation between variables, with 95% confidence intervals;
    if plot_regression:
        for i, (kind, g) in enumerate(data.groupby("kind", sort=False)):
            data_tmp = g.copy()
            # We fit a straight line on the log of the relative performance, as the scaling is exponential.
            # Then, the real prediction is 10**prediction;
            regr = linear_model.LinearRegression()
            regr.fit(data_tmp["year"].values.reshape(-1, 1), np.log10(data_tmp["performance"].values.reshape(-1, 1)))
            data_tmp["prediction"] = np.power(10, regr.predict(data_tmp["year"].values.astype(float).reshape(-1, 1)))
            ax = sns.lineplot(
                x=[data_tmp["year"].iloc[0], data_tmp["year"].iloc[-1]],
                y=[data_tmp["prediction"].iloc[0], data_tmp["prediction"].iloc[-1]],
                color=PALETTE[i],
                ax=ax,
                alpha=0.5,
                linewidth=6,
            )

            # Use the regression line to obtain the slope over 2 and 10 years;
            slope = (np.log10(data_tmp["prediction"].iloc[-1]) - np.log10(data_tmp["prediction"].iloc[0])) / (
                (data_tmp["year"].iloc[-1] - data_tmp["year"].iloc[0]).days / 365
            )
            slope_2_years = 10 ** (slope * 2)
            slope_20_years = 10 ** (slope * 20)
            kind_increase[kind] = (slope_2_years, slope_20_years)

    #####################
    # Add labels ########
    #####################

    # Associate a color to each kind of hardware (compute, memory, interconnection)
    def get_color(c: str) -> tuple[float, float, float]:
        _, saturation, brightness = colors.rgb_to_hsv(colors.to_rgb(c))
        # Make the color darker, to use it for text;
        r, g, b = sns.set_hls_values(c, l=brightness * 0.6, s=saturation * 0.7)
        return r, g, b

    kind_to_col = {k: get_color(PALETTE[i]) for i, k in enumerate(data["kind"].unique())}

    data["name"] = data["name"].fillna("")
    for i, row in data.iterrows():
        label = row["name"]
        # Label-specific adjustments;
        if label:
            if label == "Pentium II Xeon":
                xytext = (5, -9)
            elif label == "PCIe 4.0":
                xytext = (5, -9)
            elif label == "Radeon Fiji":
                xytext = (-7, 5)
            elif label == "TPUv2":
                xytext = (-7, 5)
            elif row["kind"] == "interconnect":
                xytext = (0, -9)
            else:
                xytext = (0, 5)
            ax.annotate(
                label,
                xy=(row["year"], row["performance"]),
                size=7,
                xytext=xytext,
                textcoords="offset points",
                ha="center",
                color=kind_to_col[row["kind"]],
            )

    #####################
    # Style fine-tuning #
    #####################

    # Log-scale y-axis;
    plt.yscale("log")

    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(True, linewidth=0.3)

    # Set tick number and parameters on x and y axes;
    def year_formatter(x: float, pos: Optional[str] = None) -> str:
        d = num2date(x)
        if (d.year - X_LIMITS[0].year) % 3 != 0:
            return ""
        else:
            return str(d.year)

    # Ticks, showing relative performance;
    def speedup_formatter(label: float) -> str:
        if label >= 1:
            return str(int(label))
        else:
            return f"{label:.1f}"

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(FuncFormatter(year_formatter))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_formatter(lambda x, pos: speedup_formatter(x) + r"$\mathdefault{\times}$")
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="major", labelsize=7)

    # Add a fake legend with summary data.
    # We don't use a real legend as we need rows with different colors and we don't want patches on the left.
    # Also, we want the text to look justified;
    def get_kind_label(kind: str) -> str:
        return {
            "compute": "HW FLOPS",
            "memory": "DRAM BW",
            "interconnect": "Interconnect BW",
        }.get(kind, "")

    # Create a rectangle used as background;
    rectangle = {
        "facecolor": "white",
        "alpha": 1,
        "edgecolor": "#2f2f2f",
        "linewidth": 0.5,
        "pad": 2,
    }
    # Add padding to first label, to create a large rectangle that covers other labels;
    pad = " " * 59 + "\n\n"
    # Create a second rectangle, used as shadow for the legend;
    shadow = rectangle.copy()
    shadow.pop("facecolor")
    shadow.pop("edgecolor")
    shadow["color"] = "#2f2f2f"
    ax.annotate(
        get_kind_label("compute") + ":" + pad,
        xy=(0.026, 0.93),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7,
        bbox=shadow,
    )

    for i, (k, v) in enumerate(kind_increase.items()):
        # Use two annotations, to make the text look justified;
        ax.annotate(
            get_kind_label(k) + ":" + (pad if i == 0 else ""),
            xy=(0.023, 0.94 - 0.05 * i),
            xycoords="axes fraction",
            fontsize=7,
            color=kind_to_col[k],
            ha="left",
            va="top",
            bbox=rectangle if i == 0 else None,
        )
        ax.annotate(
            f"{v[1]:.0f}"
            + r"$\mathdefault{\times}$"
            + f"/20 years ({v[0]:.1f}"
            + r"$\mathdefault{\times}$"
            + "/2 years)",
            xy=(0.43, 0.941 - 0.05 * i),
            xycoords="axes fraction",
            fontsize=7,
            color=kind_to_col[k],
            ha="right",
            va="top",
        )

    # Add axes labels;
    plt.ylabel("Performance Scaling", fontsize=8)
    plt.xlabel("")

    return fig, ax


##############################
# Load data and plot #########
##############################


def load_data() -> pd.DataFrame:
    # Load data;
    data = pd.read_csv(DATA_DIR / "performance_scaling_data.csv")
    # Convert date;
    data["year"] = pd.to_datetime(data["year"], format="ISO8601")
    return data


def plot(data: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    return performance_scaling(data, set_axes_limits=True, plot_regression=True)


##############################
# Main #######################
##############################

if __name__ == "__main__":
    data = load_data()
    fig, ax = plot(data)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="performance_scaling",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
