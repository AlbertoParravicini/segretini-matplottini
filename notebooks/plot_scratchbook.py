#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:42:20 2020

This file contains a collection of common settings to setup plots 
or adjust small formatting details.
If you don't remember how to add a legend or update tick labels or change font,
this is a good place to start your search; 

@author: aparravi
"""

import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from segretini_matplottini.utils.colors import (B4, C1, C2, PALETTE_O, R1, R2,
                                                R3, R5)
from segretini_matplottini.utils.plot_utils import (hex_color_to_grayscale,
                                                    save_plot)

PLOT_DIR = (Path(__file__).parent.parent / "plots").resolve()

if __name__ == "__main__":

    ##############################
    # Set style and fonts ########
    ##############################

    # Reset all values to default, always a good practice when changing global values using "sns.set_style" and "plt.rcParams";
    plt.rcdefaults()

    # Do this at the very beginning, they are stored in the current session.
    # If using a kernel/notebook, restoring default values require restaring the kernel;

    # Specify seaborn style, and add tick marks to y and x axis (left and bottom);
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    # Set the default font to Latin Modern Demi (very readable for papers!);
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # After installing new fonts, use matplotlib.font_manager._rebuild() to make them available;

    # Hard-coded padding values, use with care for precise formatting;
    plt.rcParams["axes.titlepad"] = 25  # Title padding;
    plt.rcParams["axes.labelpad"] = 10  # Padding of axis labels;
    plt.rcParams["axes.titlesize"] = 22  # Title size, it's better to override it when adding the title;
    plt.rcParams["axes.labelsize"] = 14  # Label size of axis, you should also override it;
    plt.rcParams["hatch.linewidth"] = 0.3  # Size of hatches in bars;

    ##############################
    # Creating new plots #########
    ##############################

    # Option 1: create a new figure and axis by hand;
    # Note: this will not have the style set using sns.set_style()!
    fig, ax = plt.subplots()
    # plt.close(fig)  # Close the figure so it's not displayed;

    # Option 2: create a new figure, then create a GridSpec to create multiple subplots.
    # This has the style specified with sns.set_style()
    num_col = 2
    num_row = 3
    fig = plt.figure(figsize=(4 * num_col, 3 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    ax = fig.add_subplot(gs[1 % num_row, 1 // num_row])  # [row number, column number]
    # Other options:
    # - Create it directly using e.g. ax = sns.lineplot(...) or fig = sns.catplot(...)
    # - If adding a single subplot, you can do ax = fig.add_subplot() without GridSpec

    ##############################
    # Adjust plot margins ########
    ##############################

    # These settings affect the current figure;
    plt.subplots_adjust(
        top=0.90,  # Max is 1
        bottom=0.2,  # Min is 0
        left=0.08,  # Min is 0
        right=0.88,  # Max is 1
        hspace=0.7,  # Vertical space (height)
        wspace=0.6,  # Horizontal space (width)
    )

    # Despine plot (i.e. delete bars top and right);
    sns.despine(ax=ax, top=True, right=True)

    # I don't like despined plots;
    sns.despine(ax=ax, top=False, right=False)
    # Enable horizontal grid lines;
    ax.grid(axis="y")

    # Set the number of ticks on the y axis;
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # This also works;
    ax.yaxis.set_major_locator(plt.LinearLocator(5))

    # Set axes to log-scale;
    plt.yscale("log")
    plt.xscale("log")
    # Set back to linear scale;
    plt.yscale("linear")
    plt.xscale("linear")

    ##############################
    # Palettes ###################
    ##############################

    # Plot to visualize colors;
    sns.palplot(PALETTE_O)

    # Custom color palette ranging from color 1 to color 2, with a neutral tone in the middle, and 20 shades;
    cm = LinearSegmentedColormap.from_list("test", [B4, "#DEDEDE", R5], N=20)
    # Obtain 10 colors as:
    colors = [cm(x) for x in np.linspace(0, 1, 10)]
    sns.palplot(colors)

    # Manipulate HLS values of color;
    new_color = sns.set_hls_values("#DEDEDE", h=0.2, l=0.3, s=0.4)

    # Obtain the same palette in grayscale;
    grayscale_colors = [hex_color_to_grayscale(c) for c in colors]
    sns.palplot(grayscale_colors)

    ##############################
    # Axis ticks and labels ######
    ##############################

    # Add some random data to a plot;
    fig = plt.figure(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    ax.scatter(x, y)
    ax.grid(axis="y")

    # Manually modify ticks.
    # This sometimes doesn't work (usually, if the x-axis is continuous instead of discrete);
    xlabels = [f"{str(x._text).upper()}" for x in ax.get_xticklabels()]
    # Rotate ticks by 45 degrees, and right-align them using "anchor" for correct visualization;
    ax.set_xticklabels(labels=xlabels, rotation=45, ha="right", rotation_mode="anchor")

    # Set percentage-based tick labels;
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(100 * x)}%"))
    # Set speedup-like tick labels (e.g. 1x, 2x, ...);
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))
    # Other formatters:
    #   * https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-formatters.html
    #   * https://matplotlib.org/3.1.1/api/ticker_api.html

    # Use exponential notation for labels (requires to know the labels!);
    # ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=15)

    # Modify ticks parameters;
    ax.tick_params(axis="x", which="major", labelsize=12, pad=2)
    # If using a formatter instead of manually specifying labels as above,
    # you can also modify the tick label alignment as follows.
    # Note that ax.tick_params() does not currently allow tto specify the alignment!
    for l in ax.get_xticklabels():
        l.set_ha("right")

    # Set axis label;
    ax.set_xlabel("X Axis", fontsize=12)

    ##############################
    # Legend #####################
    ##############################

    # Create a new custom legend;
    labels = [R1, R2, R3]
    # Create "colors" for the legend;
    custom_lines = [Patch(facecolor=x, edgecolor="#2f2f2f", label=x) for x in labels]
    # Create the legend. Note that the legend is positioned using the figure percentage (in this case, top-right corner),
    #   which is better for plots with many subplots;
    # Use ax.legend(...) to create a legend relative to the current axis;
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=12)
    leg.set_title("My Custom Legend")
    leg._legend_box.align = "left"

    # Another legend, using dots instead of rectangles to denote colors;
    custom_lines = [
        Line2D([0], [0], marker="o", color="w", label="Label 1", markerfacecolor=C1, markersize=15),
        Line2D([0], [0], marker="o", color="w", label="Label 2", markerfacecolor=C2, markersize=15),
    ]
    leg = fig.legend(
        custom_lines, ["Label 1", "Label 2"], bbox_to_anchor=(1, 0.7), fontsize=14, title_fontsize=12, ncol=2
    )
    leg.set_title("My Custom Legend, 2", prop={"size": 18})
    leg._legend_box.align = "left"

    # Set legend title. Use an if-statement as we are accessing the axis legend,
    #   which might not exist in these examlples;
    if ax.get_legend():
        ax.get_legend().set_title("My Legend")
        # Manually modify legend titles;
        for t in ax.get_legend().texts:
            t.set_text(t.get_text().upper())

    ##############################
    # Set axis limits ############
    ##############################

    # Get/set axis limits.
    # Better do it after plotting, as values might be overwritten by the plotting function;
    ax.set_ylim((0, ax.get_ylim()[1]))

    ##############################
    # Annotations ################
    ##############################

    # More info: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html

    # By default, "xy" is given in "data" coordinates, useful to annotate specific values in the plot.
    # "textcoords" can be "offset points" or "offset pixels", and is used to move a little bit the text from
    #   the corresponding point;
    ax.annotate("Annotation text", xy=(100, 200), fontsize=10, textcoords="offset points", xytext=(5, 10))
    # The annotation uses percentage-based axis coordinates.
    # Useful to add titles/labels to subplots.
    # Using "figure fraction" uses percentage-based figure coordinates,
    #   useful to add precise titles to the plot;
    ax.annotate(
        "Annotation text 2",
        xy=(0, 1),
        xycoords="axes fraction",
        fontsize=14,
        textcoords="offset points",
        xytext=(-30, 20),
        horizontalalignment="left",
        verticalalignment="center",
    )

    ##############################
    # Save plot ##################
    ##############################

    save_plot(PLOT_DIR, "test.{}")
