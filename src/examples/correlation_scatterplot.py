# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:38:13 2021

@author: albyr
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gmean

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection

import sys
sys.path.append("..")
from plot_utils import *

##############################
##############################

# Color palette used for plotting;
PALETTE = [PALETTE_G[0], PALETTE_G[2]]

# Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (-0.2, 0.6)
Y_LIMITS = (-0.1, 0.3)

##############################
##############################

def correlation_scatterplot(data: pd.DataFrame,
                            plot_kde: bool=True,
                            plot_regression: bool=True,
                            set_axes_limits: bool=True,
                            highlight_negative_area: bool=True) -> (plt.Figure, plt.Axes):
    """
    Plot a detailed correlation analysis between two variables. 
    Combine a bivariate density plot, a regression plot and a scatterplot.
    This example shows how to modify low-level parameters in the regression plot 
      which are not directly exposed by the seaborn API, 
      and how to add a rotated label with the same slope as the regression line;

    Parameters
    ----------
    data : pd.DataFrame with 2 numeric columns ("estimate0" and "estimate1") with values distributed around 0.
        Axes are truncated to fit in X_LIMITS and Y_LIMITS.
        The plot can easily be adapted to support any other univariate linear regression data, however;
        
    plot_kde : if True, add a seaborn KDE plot with the bivariate density;
    plot_regression : if True, add a seaborn linear regression plot;    
    set_axes_limits : if True, limit the axes to X_LIMITS and Y_LIMITS;
    highlight_negative_area : if True, highlight the zero axes and the negative area of the plot;
    
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
    plt.rcParams['axes.labelpad'] = 5  # Padding between axis and axis label;
    plt.rcParams['xtick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['ytick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['axes.linewidth'] = 0.8  # Line width of the axis borders;
    
    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(3.4, 3.1))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.95,
                        bottom=0.15,
                        left=0.19,
                        right=0.93)  
    ax = fig.add_subplot(gs[0, 0])
    
    # Set axes limits;        
    if set_axes_limits:
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)    
    
    # Highlight the negative part of the plot and the zero axes;
    if highlight_negative_area:
        ax.axhline(0, color="#2f2f2f", linewidth=0.5, zorder=1)
        ax.axvline(0, color="#2f2f2f", linewidth=0.5, zorder=1)
        # Create a Rectangle patch to highlight the negative part. The rectangle is created as ((start_x, start_y), width, height);
        new_patch = Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), -ax.get_xlim()[0], -ax.get_ylim()[0],
                              linewidth=0, edgecolor=None, facecolor="#cccccc", alpha=0.4, zorder=1)
        ax.add_patch(new_patch)
    
    #################
    # Main plot #####
    #################
    
    # Add a density plot for the bivariate distribution;
    if plot_kde:
        ax = sns.kdeplot(x="estimate0", y="estimate1", data=data, levels=5, color=PALETTE_O[3],
                          linewidths=1, fill=True, alpha=0.5, zorder=2)
    
    # Add a regression plot to highlight the correlation between variables, with 95% confidence intervals;
    if plot_regression:
        ax = sns.regplot(x="estimate0", y="estimate1", data=data, color=PALETTE[1], ax=ax, truncate=False,
                         scatter=False, ci=95, line_kws={"linewidth": 0.8, "linestyle": "--", "zorder": 3})
        # Update regression confidence intervals, 
        #   to set the confidence bands as semi-transparent and change style and colors of borders;
        plt.setp(ax.collections[-1], facecolor="w", edgecolor=PALETTE[1], alpha=0.6, linestyles="--", zorder=3)
    
    # Add a scatterplot for individual elements of the dataset, and change color based on statistical significance;
    ax = sns.scatterplot(x="estimate0", y="estimate1", hue="significant", palette=PALETTE, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=4)
    ax.legend_.remove()  # Hack to remove legend;
    
    #####################
    # Style fine-tuning #
    #####################
    
    # Add a label with the R^2 correlation factor. First, obtain coefficients from the linear regression;
    if plot_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(data["estimate0"], data["estimate1"])
        angle = slope * 2 * 180 / np.pi  # Convert slope angle from radians to degrees;
        # Add label with Latex Math font, at the right angle;
        ax.annotate(r"$\mathdefault{R^2=" + f"{r_value:.2f}}}$", xy=(0.47, 0.42 * slope + intercept),
                    rotation=angle, fontsize=8, ha="center", color="#2f2f2f")
                 
    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.5)
    ax.xaxis.grid(True, linewidth=0.5)

    # Ticks, showing speedup percentage (0% is no speedup);
    ax.yaxis.set_major_locator(plt.LinearLocator(9))
    ax.set_yticklabels(labels=[f"{l * 100:.0f}%" for l in ax.get_yticks()], ha="right", fontsize=8)
    ax.xaxis.set_major_locator(plt.LinearLocator(9))
    ax.set_xticklabels(labels=[f"{l * 100:.0f}%" for l in ax.get_xticks()], ha="center", fontsize=8)

    # Add legend;
    labels = ["True", "False"]
    custom_lines = [Patch(facecolor=PALETTE[::-1][i], edgecolor="#2f2f2f", label=l) for i, l in enumerate(labels)]
    leg = ax.legend(custom_lines, labels, bbox_to_anchor=(1, 0.0), fontsize=6, ncol=len(labels),
                    loc="lower right", handletextpad=0.3, columnspacing=1,)
    leg.set_title("Practically Significant", prop={"size": 6})
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    # Add axes labels;
    plt.ylabel("Speedup estimate, method B (%)", fontsize=8)
    plt.xlabel("Speedup estimate, method A (%)", fontsize=8)
    
    return fig, ax

##############################
##############################

if __name__ == "__main__":
    
    # Load data;
    data = pd.read_csv("../../data/correlation_scatterplot_data.csv")   
    
    # Remove outliers present in the dataset;
    data = data[data["estimate0"] < 1]
    data = data[data["estimate1"] < 1]
    
    # Create the plot;
    fig, ax = correlation_scatterplot(data)   
    
    # Save the plot;
    save_plot("../../plots", "correlation_scatterplot.{}")  
    

