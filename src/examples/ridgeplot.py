#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:14:53 2020

Example of Ridge plot, inspired by https://seaborn.pydata.org/examples/kde_ridgeplot.html

@author: aparravi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as st
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import os

import sys
sys.path.append("..")
from plot_utils import *

PALETTE = [COLORS["peach2"], COLORS["g1"]]


def load_data(path):
    res = pd.read_csv(path, sep=", ") 
    
    res = remove_outliers_df_grouped(res, "exec_time_u_k_us", ["opt_level", "simplify", "num_elements"], debug=False)
    res = remove_outliers_df_grouped(res, "exec_time_m_k_us", ["opt_level", "simplify", "num_elements"], debug=False)
    res = remove_outliers_df_grouped(res, "exec_time_u_us", ["opt_level", "simplify", "num_elements"], debug=False)
    res = remove_outliers_df_grouped(res, "exec_time_m_us", ["opt_level", "simplify", "num_elements"], debug=False)
    
    res["speedup"] = 1
    res["speedup_k"] = 1
    compute_speedup(res, "exec_time_u_k_us", "exec_time_m_k_us", "speedup_k")
    compute_speedup(res, "exec_time_u_us", "exec_time_m_us", "speedup")
    
    res["time_u_k_ms"] = res["exec_time_u_k_us"] / 1000 
    res["time_m_k_ms"] = res["exec_time_m_k_us"] / 1000
    res["time_u_ms"] = res["exec_time_u_us"] / 1000
    res["time_m_ms"] = res["exec_time_m_us"] / 1000
    
    return res


def ridgeplot(res):
    # Plotting setup;
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    
    x_lim = (0.7, 1.3)
    
    # Initializethe plot;
    g = sns.FacetGrid(res_tot, row="kernel_group", hue="kernel", aspect=8, height=1, palette=["#2f2f2f"], sharey=False, col="group")

    # Plot a vertical line corresponding to speedup = 1;
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=0.8)         
    # Plot the densities. Plot them twice as the second time we plot just the black contour.
    # "cut" removes values above the threshold; clip=x_lim avoids plotting values outside the margins;
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, clip=x_lim, shade=True, alpha=0.8, lw=1, bw=0.25, color=PALETTE[0], zorder=2, cut=10)  
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, clip=x_lim, shade=True, alpha=0.8, lw=1, bw=0.25, color=PALETTE[1], zorder=3, cut=10)
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, clip=x_lim, color="#5f5f5f", lw=1.5, bw=0.25, zorder=2, cut=10)
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, clip=x_lim, color="#5f5f5f", lw=1.5, bw=0.25, zorder=3, cut=10)
    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=1, clip_on=False, zorder=4)
    
    # Fix the horizontal axes so that they are in the specified range (x_lim);
    def set_x_width(label="", color="#2f2f2f"):
        ax = plt.gca()
        ax.set_xlim(left=x_lim[0], right=x_lim[1])
    g.map(set_x_width)
    
    # Plot the name of each plot;
    def label(x, label, color="#2f2f2f"):
        ax = plt.gca()
        ax.text(0, 0.15, label, color=color, ha="left", va="center", transform=ax.transAxes, fontsize=18)      
    g.map(label, "kernel")
    
    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(top=0.96,
                      bottom=0.17,
                      right=0.98,
                      left=0.02,
                      hspace=-0.20,
                      wspace=0.1)
    
    # Titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    g.fig.get_axes()[-1].set_xlabel("Relative Execution Time", fontsize=18)
    g.fig.get_axes()[-2].set_xlabel("Relative Execution Time", fontsize=18)
    
    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(100 * x)}%"
    g.fig.get_axes()[-1].xaxis.set_major_formatter(major_formatter)
    g.fig.get_axes()[-2].xaxis.set_major_formatter(major_formatter)
    g.set(yticks=[])
    g.fig.get_axes()[-1].tick_params(axis='x', which='major', labelsize=16)
    g.fig.get_axes()[-2].tick_params(axis='x', which='major', labelsize=16)
    g.despine(bottom=True, left=True)
    for tic in g.fig.get_axes()[-1].xaxis.get_major_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    for tic in g.fig.get_axes()[-2].xaxis.get_major_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    # Add custom legend;
    labels = ["Type 1", "Type 2"]
    custom_lines = [Patch(facecolor=PALETTE[0], edgecolor="#2f2f2f", label=labels[0]),
                    Patch(facecolor=PALETTE[1], edgecolor="#2f2f2f", label=labels[1])] 
      
    leg = g.fig.legend(custom_lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=17, ncol=2, handletextpad=0.5, columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    return g


if __name__ == "__main__":
    
    ##################################
    # Load data ######################
    ##################################
    
    RES_FOLDER = "../../data/ridgeplot_data"
    
    KERNELS =  ["axpy", "dot_product", "convolution", "mmul", "autocov", "hotspot", "hotspot3d",
            "backprop", "backprop2", "bfs", "pr", "nested", "gaussian",
            "histogram", "lud", "needle"]
    
    # Load all the datasets and keep only the data relevant to visualization;
    res_list = []
    for i, k in enumerate(KERNELS):
        for f in os.listdir(RES_FOLDER):
            if os.path.splitext(f)[0] == k:
                temp_res= load_data(os.path.join(RES_FOLDER, f))
                temp_res = temp_res[(temp_res["simplify"] == "simplify_accesses") & (temp_res["opt_level"] == "O2") & (temp_res["num_elements"] == max(temp_res["num_elements"]))].reset_index()  
                temp_res["kernel"] = f"B{i}"
                 # Normalize using the unmodified kernel time median;
                temp_res["time_m_k_ms"] /= np.median(temp_res["time_u_k_ms"])
                temp_res["time_u_k_ms"] /= np.median(temp_res["time_u_k_ms"])
                res_list += [temp_res]
                
    # As we are plotting on 2 columns, we need to explicitely assign the column to each plot;
    num_col = 2
    for i in range(num_col):
        # Process only a chunk of the datasets for each column;
        start = i * len(res_list) // num_col  
        end = (i + 1) * len(res_list) // num_col
        for j in range(start, end):
            res_list[j]["group"] = i  # Assign column;
            res_list[j]["kernel_group"] = j % (len(res_list) // num_col)  # Assign row;
        
    res_tot = pd.concat(res_list)
    
    ##################################
    # Plotting #######################
    ##################################  
    
    g = ridgeplot(res_tot)
    # Save the plot;
    save_plot("../../plots", "ridgeplot.{}") 
   