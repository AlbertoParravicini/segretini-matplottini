#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:14:53 2020

@author: aparravi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as st
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from plot_utils import compute_speedup, COLORS
import os


def load_data(path):
    res = pd.read_csv(path, sep=", ") 
    
    res["speedup"] = 1
    res["speedup_k"] = 1
    compute_speedup(res, "exec_time_u_k_us", "exec_time_m_k_us", "speedup_k")
    compute_speedup(res, "exec_time_u_us", "exec_time_m_us", "speedup")
    
    res["time_u_k_ms"] = res["exec_time_u_k_us"] / 1000 
    res["time_m_k_ms"] = res["exec_time_m_k_us"] / 1000
    res["time_u_ms"] = res["exec_time_u_us"] / 1000
    res["time_m_ms"] = res["exec_time_m_us"] / 1000
    
    # Keep only the data subset we care about in the plot;
    return res[(res["simplify"] == "simplify_accesses") & (res["opt_level"] == "O2") & (res["num_elements"] == max(res["num_elements"]))].reset_index()  


def remove_outliers(data, threshold=3):
    sizes = set(data["num_elements"])
    types_k = ["time_m_k_ms", "time_u_k_ms"]
    types = ["time_m_ms", "time_u_ms"]
    
    data_list = []
    for s in sizes:
        temp_data = data[data["num_elements"] == s]
        for t in types_k:
            temp_data = temp_data[st.zscore(temp_data[t]) < 3]
        for t in types:
            temp_data = temp_data[st.zscore(temp_data[t]) < 3]
        data_list += [temp_data]
    return pd.concat(data_list)


def ridgeplot(res, names):
    # Plotting setup;
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    
    # Initializethe plot;
    g = sns.FacetGrid(res_tot, row="kernel_group", hue="kernel", aspect=7, height=1, palette=["#2f2f2f"], sharey=False, col="group")

    # Plot a vertical line corresponding to speedup = 1;
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=0.5)         
    # Plot the densities. Plot them twice as the second time we plot just the white contour;                                                                                 
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, shade=True, alpha=0.6, lw=1, bw=0.02, color=COLORS["b1"], zorder=2, cut=10)  
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, shade=True, alpha=0.6, lw=1, bw=0.02, color=COLORS["r1"], zorder=3, cut=10)
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, color="w", lw=1.5, bw=0.02, zorder=2, cut=10)
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, color="w", lw=1.5, bw=0.02, zorder=3, cut=10)
    # Plot the horizontal line below the densities;
    g.map(plt.axhline, y=0, lw=1, clip_on=False, zorder=4)
    
    # Fix the horizontal axes so that they are between 0.5 and 1.25;
    def set_x_width(label="", color="#2f2f2f"):
        ax = plt.gca()
        ax.set_xlim(left=0.5, right=1.25)
    g.map(set_x_width)
    
    # Plot the name of each plot;
    def label(x, label, color="#2f2f2f"):
        ax = plt.gca()
        ax.text(0, 0.15, label, color=color, ha="left", va="center", transform=ax.transAxes, fontsize=14)      
    g.map(label, "kernel")
    
    # Fix the borders. This must be done here as the previous operations update the default values;
    g.fig.subplots_adjust(top=0.83,
                      bottom=0.15,
                      right=0.95,
                      left=0.05,
                      hspace=-0.20,
                      wspace=0.1)
    
    # Titles and labels;
    g.set_titles("")
    g.set(xlabel=None)
    g.fig.get_axes()[-1].set_xlabel("Relative Kernel Execution Time", fontsize=15)
    g.fig.get_axes()[-2].set_xlabel("Relative Kernel Execution Time", fontsize=15)
    
    # Write the x-axis tick labels using percentages;
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(100 * x)}%"
    g.fig.get_axes()[-1].xaxis.set_major_formatter(major_formatter)
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Add custom legend;
    custom_lines = [Patch(facecolor=COLORS["b1"], edgecolor="#2f2f2f", label="Manually Modified"),
                    Patch(facecolor=COLORS["r1"], edgecolor="#2f2f2f", label="Automatically Modified"),
                    ]
    leg = g.fig.legend(custom_lines, ["Baseline", "Automatically Modified"],
                             bbox_to_anchor=(0.97, 0.98), fontsize=15)
    leg.set_title("Kernel Type", prop={"size": 15})
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    # Main plot title;
    g.fig.suptitle("Kernel Relative Exec. Time Distribution,\nO2 Opt. Level", ha="left", x=0.05, y=0.95, fontsize=18)
    
    return g


if __name__ == "__main__":
    
    ##################################
    # Load data ######################
    ##################################
    
    res_folder = "../data/ridgeplot_data"
    
    # Load all the datasets;
    res_axpy = load_data(os.path.join(res_folder, "axpy.csv"))
    res_dp = load_data(os.path.join(res_folder, "dot_product.csv"))
    res_conv = load_data(os.path.join(res_folder, "convolution.csv"))
    res_mmul = load_data(os.path.join(res_folder, "mmul.csv"))
    res_autocov = load_data(os.path.join(res_folder, "autocov.csv"))
    res_hotspot = load_data(os.path.join(res_folder, "hotspot.csv"))
    res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop.csv"))
    res_bb2 = load_data(os.path.join(res_folder, "backprop2.csv"))
    res_bfs = load_data(os.path.join(res_folder, "bfs.csv"))
    res_pr = load_data(os.path.join(res_folder, "pr.csv"))  
    res_gaussian = load_data(os.path.join(res_folder, "gaussian.csv"))
    res_histogram = load_data(os.path.join(res_folder, "histogram.csv"))
    res_lud = load_data(os.path.join(res_folder, "lud.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle.csv"))
    res_nested = load_data(os.path.join(res_folder, "nested.csv"))
        
    res_list = [res_axpy, res_dp, res_conv, res_autocov, res_hotspot3d, res_bb, res_bfs, res_pr, res_nested, res_mmul, res_hotspot, 
                 res_bb2, res_gaussian, res_histogram, res_lud, res_needle]
    
    # Names used in the plot for each dataset;
    names = ["Axpy", "Dot Product", "Convolution 1D", "Auto-covariance", "Hotspot 3D",
             "NN - Forward Pass", "BFS", "PageRank", "Nested Loops", "Matrix Multiplication", "Hotspot",  "NN - Backpropagation",  "Gaussian Elimination",
             "Histogram", "LU Decomposition", "Needleman-Wunsch"]
    
    # Remove outliers from datasets;
    res_list = [remove_outliers(res) for res in res_list]
    
    # As we are plotting on 2 columns, we need to explicitely assign the column to each plot;
    num_col = 2
    for i in range(num_col):
        # Process only a chunk of the datasets for each column;
        start = i * len(res_list) // num_col  
        end = (i + 1) * len(res_list) // num_col
        for j in range(start, end):
            res_list[j]["group"] = i  # Assign column;
            res_list[j]["kernel_group"] = j % (len(res_list) // num_col)  # Assign row;
        
    # Create a unique table by merging the datasets;
    res_list_filtered = []
    for i, res in enumerate(res_list):
        res["kernel"] = names[i]
         # Normalize using the unmodified kernel time median;
        res["time_m_k_ms"] /= np.median(res["time_u_k_ms"])
        res["time_u_k_ms"] /= np.median(res["time_u_k_ms"])
        res_list_filtered += [res]
    res_tot = pd.concat(res_list_filtered)
    
    # Plotting;
    g = ridgeplot(res_tot, names)
    # Save the plot;
    plt.savefig("../plots/ridgeplot.pdf")
    
   