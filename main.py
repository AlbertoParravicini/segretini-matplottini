#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:42:20 2020

@author: aparravi
"""

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from plot_utils import *

# Define some colors for later use.
# Tool to create paletters: https://color.adobe.com/create
# Guide to make nice palettes: https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
COLORS = dict(
    c1 = "#b1494a",
    c2 = "#256482",
    c3 = "#2f9c5a",
    c4 = "#28464f",
    
    r4 = "#CE1922",
    r3 = "#F41922",
    r2 = "#FA3A51",
    r1 = "#FA4D4A",
    r5 = "#F07B71",
    r6 = "#F0A694",
    
    b1 = "#97E6DB",
    b2 = "#C6E6DB",
    b3 = "#CEF0E4",
    b4 = "#9CCFC4",
    b5 = "#AEDBF2",
    b6 = "#B0E6DB",
    b7 = "#B6FCDA",
    b8 = "#7bd490",
    
    y1 = "#FFA728",
    y2 = "#FF9642",
    y3 = "#FFAB69",
    
    bt1 = "#55819E",
    bt2 = "#538F6F",
    )


#%%
if __name__ == "__main__":
    
    ##############################
    # Set style and fonts ########
    ##############################
    
    # Do this at the very beginning, they are stored in the current session.
    # If using a kernel/notebook, restoring default values require restaring the kernel;
    sns.set_style("white")
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    
    # Hard-coded padding values, use with care for precise formatting;
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    # After installing new fonts, use matplotlib.font_manager._rebuild() to make them available;

    ##############################
    # Creating new plots #########
    ##############################
    
    # Option 1: create a new figure and axis by hand;
    fig, ax = plt.subplots()
    
    # Option 2: create a new figure, then create a GridSpec to create multiple subplots;
    num_col = 2
    num_row = 3
    fig = plt.figure(figsize=(4 * num_col, 3 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    ax = fig.add_subplot(gs[1 % num_row, 1 // num_row])
    
    # Other options: create it directly using e.g. ax = sns.lineplot(...) or fig = sns.catplot(...)
    
    ##############################
    # Adjust plot margins ########
    ##############################
    
    # These settings affect the current figure;
    plt.subplots_adjust(top=0.90,  # Max is 1
                        bottom=0.04,  # Min is 0 
                        left=0.08,  # Min is 0
                        right=0.88,  # Max is 1
                        hspace=0.7,  # Vertical space (height)
                        wspace=0.6)  # Horizontal space (width)
    
    # Despine plot;
    sns.despine(ax=ax, top=True, right=True)
    
    # Enable horizontal grid lines;
    ax.grid(axis='y')
    
    #%%
    
    ##############################
    # Palettes ###################
    ##############################
    
    # Plot to visualize colors;
    sns.palplot(COLORS.values())
    
    # Custom color palette ranging from color 1 to color 2, with a neutral tone in the middle, and 20 shades;
    cm = LinearSegmentedColormap.from_list("test", [COLORS["b4"], "#DEDEDE", COLORS["r5"]], N=20)
    # Obtain 10 colors as:
    colors = [cm(x) for x in np.linspace(0, 1, 10)]
    sns.palplot(colors)
    
    #%%
    
    ##############################
    # Axis ticks and labels ######
    ##############################
    
    # Add some random data to a plot;
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    ax.scatter(x, y)
    
    # Manually modify ticks. 
    # This sometimes doesn't work (usually, if the x-axis is continuous instead of discrete);
    xlabels = [f"{str(x).upper()}" for x in ax.get_xticklabels()]
    # Rotate ticks by 45 degrees, and right-align them for correct visualization;
    ax.set_xticklabels(labels=xlabels, rotation=45, ha="right")

    # Set percentage-based tick labels;    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(100 * x)}%"))
    # Set speedup-like tick labels (e.g. 1x, 2x, ...);
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))
    # Other formatters: 
    #   * https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-formatters.html
    #   * https://matplotlib.org/3.1.1/api/ticker_api.html

    # Modify ticks parameters;
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    # Set axis label;
    ax.set_xlabel("X Axis", fontsize=12)
    
    #%%
    
    ##############################
    # Legend #####################
    ##############################
        
    # Create a new custom legend;    
    labels = ["r1", "r2", "r3"]
    # Create "colors" for the legend;
    custom_lines = [Patch(facecolor=COLORS[x], edgecolor="#2f2f2f", label=x) for x in labels]    
    # Create the legend. Note that the legend is positioned using the figure percentage (in this case, top-right corner),
    #   which is better for plots with many subplots;
    # Use ax.legend(...) to create a legend relative to the current axis;
    leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=12)
    leg.set_title("My Custom Legend")
    leg._legend_box.align = "left"
    
    # Another legend, using dots instead of rectangles to denote colors;
    custom_lines = [Line2D([0], [0], marker='o', color="w", label="Label 1", markerfacecolor=COLORS["c1"], markersize=15),
                    Line2D([0], [0], marker='o', color="w", label="Label 2", markerfacecolor=COLORS["c2"], markersize=15)]    
    leg = fig.legend(custom_lines, ["Label 1", "Label 2"], bbox_to_anchor=(1, 0.7), fontsize=14, title_fontsize=12)
    leg.set_title("My Custom Legend, 2")
    leg._legend_box.align = "left"
    
    # Set legend title. Use an if-statement as we are accessing the axis legend,
    #   which might not exist in these examlples;
    if ax.get_legend():
        ax.get_legend().set_title("My Legend")
        # Manually modify legend titles;
        for t in ax.get_legend().texts:
            t.set_text(t.get_text().upper())
            
    #%%

    ##############################
    # Set axis limits ############
    ##############################       
            
    # Get/set axis limits. 
    # Better do it after plotting, as values might be overwritten by the plotting function;
    ax.set_ylim((0, ax.get_ylim()[1]))
    
    
    #%% 
    
    ##############################
    # Annotations ################
    ############################## 
    
    # More info: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html
    
    # By default, "xy" is given in "data" coordinates, useful to annotate specific values in the plot.
    # "textcoords" can be "offset points" or "offset pixels", and is used to move a little bit the text from 
    #   the corresponding point;
    ax.annotate("Annotation text",
                xy=(100, 200), fontsize=10,
                textcoords="offset points", xytext=(5, 10))
    # The annotation uses percentage-based axis coordinates. 
    # Useful to add titles/labels to subplots. 
    # Using "figure fraction" uses percentage-based figure coordinates, 
    #   useful to add precise titles to the plot;
    ax.annotate("Annotation text 2",
                xy=(0, 1), xycoords="axes fraction", fontsize=14, textcoords="offset points", xytext=(-30, 20),
                horizontalalignment="left", verticalalignment="center")
    
    #%% 
    
    ##############################
    # Save plot ##################
    ############################## 
    
    save_folder = "plots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    # Use "pdf" or "png". The "dpi" setting is only relevant for "png";
    extension = "pdf"
    plt.savefig(f"plots/test.{extension}", dpi=200)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    