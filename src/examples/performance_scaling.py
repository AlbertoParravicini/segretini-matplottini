# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:53:24 2021

@author: albyr
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.dates import YearLocator, MonthLocator, num2date
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from sklearn import linear_model

import sys
sys.path.append("..")
from plot_utils import *

##############################
##############################

# # Color palette used for plotting;
PALETTE = ["#F0B390", "#90D6B4", "#7BB7BA"]
MARKERS = ["o", "D", "X"]

# # Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (datetime(year=1996, month=1, day=1), datetime(year=2022, month=1, day=1))
Y_LIMITS = (0.1, 5 * 1e6)

##############################
##############################

def performance_scaling(data: pd.DataFrame,
                        set_axes_limits: bool=True,
                        plot_regression: bool=True) -> (plt.Figure, plt.Axes):
    """
    Parameters
    ----------
    data : pd.DataFrame with 6 columns:
        "year",
        "performance",
        "kind" ∈ ["compute", "memory", "interconnect"],
        "name" (label shown in the plot, it can be empty),
        "base" (base value used for speedup, it can be empty),
        "comment" (e.g. data source or non-used label, it can be empty).

    Returns
    -------
    fig : matplotlib figure containing the plot
    ax : matplotlib axis containing the plot
    """
    
    ##############
    # Plot setup #
    ##############
    
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Setup general plotting settings;
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.labelpad'] = 0  # Padding between axis and axis label;
    plt.rcParams['xtick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['ytick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['axes.linewidth'] = 0.8  # Line width of the axis borders;
    
    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(6, 2.5))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.98,
                        bottom=0.1,
                        left=0.12,
                        right=0.99)  
    ax = fig.add_subplot(gs[0, 0])
    
    # Set axes limits;        
    if set_axes_limits:
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
    
    #################
    # Main plot #####
    #################    

    # Measure performance increase over 20 and 2 years;
    kind_increase = {}      
    
    # Add a scatterplot for individual elements of the dataset, and change color based on hardware type;
    ax = sns.scatterplot(x="year", y="performance", hue="kind", style="kind", palette=PALETTE, markers=MARKERS, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=4)

    # Add a regression plot to highlight the correlation between variables, with 95% confidence intervals;
    if plot_regression:
        for i, (kind, g) in enumerate(data.groupby("kind", sort=False)):            
            data_tmp = g.copy()
            # We fit a straight line on the log of the relative performance, as the scaling is exponential.
            # Then, the real prediction is 10**prediction;
            regr = linear_model.LinearRegression()
            regr.fit(data_tmp["year"].values.reshape(-1, 1), np.log10(data_tmp["performance"].values.reshape(-1, 1)))
            data_tmp["prediction"] = np.power(10, regr.predict(data_tmp["year"].values.astype(float).reshape(-1, 1)))
            ax = sns.lineplot(x=[data_tmp["year"].iloc[0], data_tmp["year"].iloc[-1]],
                              y=[data_tmp["prediction"].iloc[0], data_tmp["prediction"].iloc[-1]],
                              color=PALETTE[i], ax=ax, alpha=0.5, linewidth=6)
            
            # Use the regression line to obtain the slope over 2 and 10 years;
            slope = (np.log10(data_tmp["prediction"].iloc[-1]) - np.log10(data_tmp["prediction"].iloc[0])) / ((data_tmp["year"].iloc[-1] - data_tmp["year"].iloc[0]).days / 365)
            slope_2_years = 10**(slope * 2)
            slope_20_years = 10**(slope * 20)
            kind_increase[kind] = (slope_2_years, slope_20_years)
    ax.legend_.remove()  # Hack to remove legend;

    #####################
    # Add labels ########
    #####################
    
    # Associate a color to each kind of hardware (compute, memory, interconnection)
    def get_color(c):  # Make the color darker, to use it for text;
        hue, saturation, brightness = colors.rgb_to_hsv(colors.to_rgb(c))
        return sns.set_hls_values(c, l=brightness * 0.6, s=saturation * 0.7)
    kind_to_col = {k: get_color(PALETTE[i]) for i, k in enumerate(data["kind"].unique())}
    
    data["name"] = data["name"].fillna("")
    for i, row in data.iterrows():
        label = row["name"]
        # Label-specific adjustments;
        if label:
            if label ==  "Pentium II Xeon":
                xytext = (5, -9)
            elif label ==  "PCIe 4.0":
                xytext = (5, -9)
            elif label ==  "Radeon Fiji":
                xytext = (-7, 5)
            elif label ==  "TPUv2":
                xytext = (-7, 5)
            elif row["kind"] == "interconnect":
                xytext = (0, -9)
            else:
                xytext = (0, 5)
            ax.annotate(label, xy=(row["year"], row["performance"]), size=7, xytext=xytext,
                        textcoords="offset points", ha="center", color=kind_to_col[row["kind"]])
    
    #####################
    # Style fine-tuning #
    #####################
    
    # Log-scale y-axis;
    plt.yscale("log")
    
    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(True, linewidth=0.3)
    
    # Set tick number and parameters on x and y axes;
    def year_formatter(x, pos=None):
        d = num2date(x)
        if (d.year - X_LIMITS[0].year) % 3 != 0:
            return ""
        else:
            return d.year
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(FuncFormatter(year_formatter))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;
    
    # Ticks, showing relative performance;
    def format_speedup(l):
        if l >= 1:
            return str(int(l))
        else:
            return f"{l:.1f}"
    ax.set_yticklabels(labels=[format_speedup(l) + r"$\mathdefault{\times}$" for l in ax.get_yticks()], ha="right", fontsize=7)
 
    # Add a fake legend with summary data.
    # We don't use a real legend as we need rows with different colors and we don't want patches on the left.
    # Also, we want the text to look justified.
    def get_kind_label(k):
        kind_name = ""
        if k == "compute":
            kind_name = "HW FLOPS"
        elif k == "memory":
            kind_name = "DRAM BW"
        else:
            kind_name = "Interconnect BW"
        return kind_name
    # Create a rectangle used as background;
    rectangle = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#B8B8B8", "linewidth": 0.5, "pad": 0.5}
    for i, (k, v) in enumerate(kind_increase.items()):
        pad = " " * 48 + "\n\n"  # Add padding to first label, to create a large rectangle that covers other labels; 
        # Use two annotations, to make the text look justified;
        ax.annotate(get_kind_label(k) + ":" + (pad if i == 0 else ""), xy=(0.023, 0.94 - 0.05 * i),
                    xycoords="axes fraction", fontsize=7, color=kind_to_col[k], ha="left", va="top", bbox=rectangle if i == 0 else None)
        ax.annotate(f"{v[1]:.0f}" + r"$\mathdefault{\times}$" + f"/20 years ({v[0]:.1f}" + r"$\mathdefault{\times}$"+ "/2 years)",
                    xy=(0.43, 0.941 - 0.05 * i), xycoords="axes fraction", fontsize=7, color=kind_to_col[k], ha="right", va="top")
        
    # Add axes labels;
    plt.ylabel("Performance Scaling", fontsize=8)
    plt.xlabel(None)
    
    return fig, ax

##############################
##############################

if __name__ == "__main__":
    
    # Load data;
    data = pd.read_csv("../../data/performance_scaling.csv")   
    # Convert date;
    data["year"] = pd.to_datetime(data["year"], format='%Y-%m')

    # Create the plot;
    fig, ax = performance_scaling(data)   
    
    # Save the plot;
    save_plot("../../plots", "performance_scaling.{}")  
    
