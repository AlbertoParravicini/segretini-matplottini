#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:06:01 2020
@author: aparravi
"""

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os
from matplotlib.colors import rgb_to_hsv, to_rgb, to_hex, hsv_to_rgb
from scipy.stats.mstats import gmean
import scipy.stats
from functools import reduce
from pathlib import Path

##############################
# Colors #####################
##############################

# Define some colors for later use.
# Tool to create paletters: https://color.adobe.com/create
# Guide to make nice palettes: https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/
COLORS = dict(

    c1 = "#b1494a",
    c2 = "#256482",
    c3 = "#2f9c5a",
    c4 = "#28464f",
    
    r1 = "#FA4D4A",
    r2 = "#FA3A51",
    r3 = "#F41922",
    r4 = "#CE1922",
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
    
    # Another blue-green palette;
    bb0 = "#FFA685",
    bb1 = "#75B0A2",
    bb2 = "#CEF0E4",  # Same as b3; 
    bb3 = "#B6FCDA",  # Same as b7;
    bb4 = "#7ED7B8",
    bb5 = "#7BD490",
    
    y1 = "#FFA728",
    y2 = "#FF9642",
    y3 = "#FFAB69",
    y4 = "#E6D955",
    
    peach1 = "#FF9868",
    peach2 = "#fab086",
    
    g1 = "#A2F2B1",
    g2 = "#A5E6C6",
    
    bt1 = "#55819E",
    bt2 = "#538F6F",
    blue_klein = "#002fa7",
    )

# Pastel-tint palette from dark blue to red;
PALETTE_1 = ["#4C9389", "#60b1b0", "#8fd4d4", "#9BD198", "#EFE8AC", "#f9af85", "#f59191"]

# Palette of green tones, light to dark;
PALETTE_G = ["#AEE69C", "#96DE9B", "#73BD8E", "#66B784", "#469E94", "#3A857C", "#34756E"] 
PALETTE_G2 = ["#D5FFC7", "#9AE39F", "#73BD8E", "#5EA889", "#4F9E8C", "#3C8585"]
PALETTE_G3 = ["#E7F7DF", "#B5E8B5", "#71BD8A", "#469C7F", "#257A7A"]
# Palette of orange/yellow/brown tones;
PALETTE_O = ["#FDCE7C", "#EBB25E", "#E6A37C", "#FAB187", "#E3896F", "#E37E62", "#FD867C"]
# Palette with orange baseline + green tones;
PALETTE_OG = ["#E8CFB5", "#81C798", "#55A68B", "#358787"]

##################################
# Utility functions for plotting #
##################################


def hex_color_to_grayscale(c):
    new_c = rgb_to_hsv(to_rgb(c))
    new_c[1] = 0  # Set saturation to 0;
    return to_hex(hsv_to_rgb(new_c))


def get_exp_label(val, prefix="", decimal_remaining_val: bool=False, integer_mantissa=True, from_string=True, decimal_places=2) -> str: 
    """
    :param val: numeric label to format
    :return: label formatted in scientific notation
    
    Format a label in scientific notation, using Latex math font.
    For example, 10000 -> 10^4;
    """
    if not from_string:
        # Get the power of 10
        exp_val = 0
        remaining_val = int(val)
        while ((remaining_val % 10 == 0 if integer_mantissa else remaining_val > 1) and remaining_val > 0):
            exp_val += 1
            remaining_val = remaining_val // 10
        if remaining_val > 1 and exp_val >= 1:
            if decimal_remaining_val and remaining_val > 0:
                return r"$\mathdefault{" + prefix + str(remaining_val / 10) + r"\!·\!{10}^{" + str(exp_val + 1) + r"}}$"
            else:
                return r"$\mathdefault{" + prefix + str(remaining_val) + r"\!·\!{10}^{" + str(exp_val) + r"}}$"
        elif remaining_val > 1 and exp_val == 0:
            print(val)
            return r"$\mathdefault{" + prefix + str(val) + r"}$"
        else:
            return r"$\mathdefault{" + prefix + r"{10}^{" + str(exp_val) + r"}}$"
    else:
        string = "{:.{prec}E}".format(val, prec=decimal_places)
        decimal_part = float(string.split("E")[0])
        sign = string.split("E")[1][0]
        exponent = int(string.split("E")[1][1:])
        if integer_mantissa:
            while (decimal_part - int(decimal_part) > 0) if val > 0 else (decimal_part - int(decimal_part) < 0):
                decimal_part *= 10
                decimal_part = float("{:.{prec}f}".format(decimal_part, prec=decimal_places))
                exponent -=1
            decimal_part = int(decimal_part)
        return r"$\mathdefault{" + prefix + str(decimal_part) + r"\!·\!{10}^{" + (sign if sign == "-" else "") + str(exponent) + r"}}$"
    

def fix_label_length(labels: list, max_length: int=20) -> list:
    """
    :param labels: a list of textual labels
    :return: a list of updated labels
    
    Ensure that all labels are shorter than a given length;
    """
    fixed_labels = []
    for l in labels:
        if len(l) <= max_length:
            fixed_labels += [l]
        else:
            fixed_labels += [l[:max_length-3] + "..."]
    return fixed_labels
    
    
def get_ci_size(x, ci=0.95, estimator=np.mean, get_raw_location: bool=False):
    """
    :param x: a sequence of numerical data, iterable
    :param ci: confidence interval to consider
    :param get_raw_location: if True, report the values of upper and lower intervals, instead of their sizes from the center
    :return: size of upper confidence interval, size of lower confidence interval, mean
    
    Compute the size of the upper confidence interval,
    i.e. the size between the top of the bar and the top of the error bar as it is generated by seaborn.
    Useful for adding labels above error bars, or to create by hand the error bars;
    """ 
    center = estimator(x)
    ci_lower, ci_upper = st.t.interval(ci, len(x) - 1, loc=center, scale=st.sem(x))
    if not get_raw_location:
        ci_upper -= center
        ci_lower =- center
    return ci_upper, ci_lower, center


def get_upper_ci_size(x, ci=0.95, estimator=np.mean):
    return get_ci_size(x, ci, estimator=estimator)[0]
    
    
def add_labels(ax: plt.Axes, labels: list=None, vertical_offsets: list=None, 
               patch_num: list=None, fontsize: int=14, rotation: int=0,
               skip_zero: bool=False, format_str: str="{:.2f}x",
               label_color: str="#2f2f2f", max_only=False,
               skip_bars: int=0, max_bars: int=None,
               skip_value: float=None, skip_threshold: float=1e-6,
               skip_nan_bars: bool=True, max_height: float=None):
    """
    :param ax: current axis, it is assumed that each ax.Patch is a bar over which we want to add a label
    :param labels: optional labels to add. If not present, add the bar height
    :param vertical_offsets: additional vertical offset for each label.
      Useful when displaying error bars (see @get_upper_ci_size), and for fine tuning
    :param patch_num: indices of patches to which we add labels, if some of them should be skipped
    :param fontsize: size of each label
    :param rotation: rotation of the labels (e.g. 90°)
    :param skip_zero: if True, don't put a label over the first bar
    :param format_str: format of each label, by default use speedup (e.g. 2.10x)
    :param label_color: hexadecimal color used for labels
    :param max_only: add only the label with highest value
    :param skip_bars: start adding labels after the specified number of bars
    :param max_bars: don't add labels after the specified bar
    :param skip_value: don't add labels equal to the specified value
    :param skip_threshold: threshold used to determine if a label is 1 or 0
    :param skip_nan_bars: if True, skip bars with NaN height when placing labels
    :param max_height: if present, place labels at this maximum specified height (e.g. the y axis limit)
        
    Used to add labels above barplots;
    """
    if not vertical_offsets:
        # 5% above each bar, by default;
        vertical_offsets = [ax.get_ylim()[1] * 0.05] * len(ax.patches)
    if not labels:
        labels = [p.get_height() for p in ax.patches]
        if max_only:
            argmax = np.argmax(labels)
    patches = []
    if not patch_num:
        patches = ax.patches
    else:
        patches = [p for i, p in enumerate(ax.patches) if i in patch_num]
    if skip_nan_bars:
        labels = [l for l in labels if not pd.isna(l)]
        patches = [p for p in patches if not pd.isna(p.get_height())]
    
    # Iterate through the list of axes' patches
    for i, p in enumerate(patches[skip_bars:max_bars]):
        if labels[i] and (i > 0 or not skip_zero) and (not max_only or i == argmax) and i < len(labels) and i < len(vertical_offsets):
            if skip_value and np.abs(labels[i] - skip_value) < skip_threshold:
                continue  # Skip labels equal to the specified value;
            height = vertical_offsets[i] + p.get_height()
            if max_height is not None and height > max_height:
                height = max_height
            ax.text(p.get_x() + p.get_width() / 2, height, format_str.format(labels[i]), 
                    fontsize=fontsize, color=label_color, ha='center', va='bottom', rotation=rotation)
        
        
def update_width(ax: plt.Axes, width: float=1):
    """
    Given an axis with a barplot, scale the width of each bar to the provided percentage,
      and align them to their center;
    """
    for i, patch in enumerate(ax.patches):
        current_width = patch.get_width()
        diff = current_width - width
        # Change the bar width
        patch.set_width(width)
        # Recenter the bar
        patch.set_x(patch.get_x() + 0.5 * diff)
        
        
def transpose_legend_labels(labels, patches, max_elements_per_row=6, default_elements_per_col=2):
    """
    Matplotlib by defaults places elements in the legend from top to bottom.
    In most cases, placing them left-to-right is more readable (you know, English is read left-to-right, not top-to-bottom)
    This function transposes the elements in the legend, allowing to set the maximum number of values you want in each row.
    
    :param labels: list of labels in the legend
    :param patches: list of patches in the legend
    :param max_elements_per_row: maximum number of legend elements per row
    :param default_elements_per_col: by default, try having default_elements_per_col elements in each col (could be more if max_elements_per_row is reached) 
    """
    elements_per_row = min(int(np.ceil(len(labels) / default_elements_per_col)), max_elements_per_row)  # Don't add too many elements per row;
    labels = np.concatenate([labels[i::elements_per_row] for i in range(elements_per_row)], axis=0)
    patches = np.concatenate([patches[i::elements_per_row] for i in range(elements_per_row)], axis=0)
    return labels, patches
        
        
def save_plot(directory: str, filename: str, figure: plt.Figure=None, date: str = "", create_date_dir: bool = True, extension: list = ["pdf", "png"], dpi=300):
    """
    :param directory: where the plot is stored
    :param filename: should be of format 'myplot_{}.{}', where the first placeholder is used for the date and the second for the extension,
        or 'myplot.{}', or 'myplot.extension'
    :param figure: a specific figure to save. If None, save the last plot that has been drawn
    :param date: date that should appear in the plot filename
    :param create_date_dir: if True, create a sub-folder with the date
    :param extension: list of extension used to store the plot
    """
    
    output_folder = os.path.join(directory, date) if create_date_dir and date else directory
    if not os.path.exists(output_folder):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
    for e in extension:
        if figure:
            figure.savefig(os.path.join(output_folder, filename.format(date, e) if date else filename.format(e)), dpi=dpi)
        else:  # Save the current plot;
            plt.savefig(os.path.join(output_folder, filename.format(date, e) if date else filename.format(e)), dpi=dpi)


####################################
# Utility functions for DataFrames #
####################################


def remove_outliers(data, sigmas: int=3):
    """
    :param data: a sequence of numerical data, iterable
    :param sigmas: number of standard deviations outside which a value is consider to be an outlier
    :return: data without outliers
    
    Filter a sequence of data by keeping only values within "sigma" standard deviations from the mean.
    This is a simple way to filter outliers, it is more useful for visualizations than for sound statistical analyses;
    """
    return data[np.abs(st.zscore(data)) < sigmas]


def remove_outliers_df(data: pd.DataFrame, column: str, reset_index: bool = True, drop_index: bool = True, sigmas: int = 3) -> pd.DataFrame:
    """
    :param data: a pd.DataFrame
    :param column: name of the column where data are filtered
    :param reset_index: if True, reset the index after filtering
    :param drop_index: if True, drop the index column after reset
    :param sigmas: number of standard deviations outside which a value is consider to be an outlier
    :return: data without outliers
    
    Filter a sequence of data by keeping only values within "sigma" standard deviations from the mean.
    This is a simple way to filter outliers, it is more useful for visualizations than for sound statistical analyses;
    """
    col = data[column]
    res = data.loc[remove_outliers(col, sigmas).index]
    if reset_index:
        res = res.reset_index(drop=drop_index)
    return res


def remove_outliers_df_grouped(data: pd.DataFrame, column: str, group: list, reset_index: bool = True, drop_index: bool = True, sigmas: int = 3, debug: bool = True) -> pd.DataFrame:
    """
    Same as "remove_outliers_df", but also filter values after divided by the group of columns specified in "group";
    """
    old_len = len(data)
    filtered = []
    for i, g in data.groupby(group, sort=False):
        filtered += [remove_outliers_df(g, column, reset_index, drop_index, sigmas)]
    new_data = pd.concat(filtered, ignore_index=True)
    if debug and (len(new_data) < old_len):
        print(f"removed {old_len - len(new_data)} outliers")
    return new_data


def remove_outliers_iqr(data, quantile: float=0.75, iqr_extension: float=1.5):
    """
    :param data: a sequence of numerical data, iterable
    :param quantile: upper quantile value used as filtering threshold. Also use (1 - quantile) as lower threshold. Should be in [0.5, 1]
    :param iqr_extension: multiple of interquantile range (iqr) used to filter data, starting from the quantiles
    :return: data without outliers
    
    Filter a sequence of data by removing outliers looking at the quantiles of the distribution.
    Find quantiles (by default, Q1 and Q3), and interquantile range (by default, Q3 - Q1), 
    and keep values in [Q1 - iqr_extension * IQR, Q3 + iqr_extension * IQR].
    This is the same range used to identify whiskers in a boxplot (e.g. in pandas and seaborn);
    """
    assert(quantile >= 0.5 and quantile <= 1)
    q1 = np.quantile(data, 1 - quantile)
    q3 = np.quantile(data, quantile)
    iqr = scipy.stats.iqr(data, rng=(100 - 100 * quantile, 100 * quantile))
    return data[(data >= q1 - iqr * q1) & (data <= q3 + iqr * q3)]


def remove_outliers_iqr_df(data: pd.DataFrame, column: str, reset_index: bool = True, drop_index: bool = True,
                           quantile: float=0.75, iqr_extension: float=1.5, debug: bool=True) -> pd.DataFrame:
    """
    :param data: a pd.DataFrame
    :param column: name of the column where data are filtered
    :param reset_index: if True, reset the index after filtering
    :param drop_index: if True, drop the index column after reset
    :param quantile: upper quantile value used as filtering threshold. Also use (1 - quantile) as lower threshold. Should be in [0.5, 1]
    :param iqr_extension: multiple of interquantile range (iqr) used to filter data, starting from the quantiles
    :return: data without outliers
    
    Filter a pd.DataFrame by removing outliers (on te specified column) looking at the quantiles of the distribution.
    Find quantiles (by default, Q1 and Q3), and interquantile range (by default, Q3 - Q1), 
    and keep values in [Q1 - iqr_extension * IQR, Q3 + iqr_extension * IQR].
    This is the same range used to identify whiskers in a boxplot (e.g. in pandas and seaborn);
    """
    old_len = len(data)
    col = data[column]
    new_data = data.loc[remove_outliers_iqr(col, quantile, iqr_extension).index]
    if reset_index:
        new_data = new_data.reset_index(drop=drop_index)
    if debug and (len(new_data) < old_len):
        print(f"removed {old_len - len(new_data)} outliers on column {column}")
    return new_data


def remove_outliers_df_iqr_grouped(data: pd.DataFrame, column: str, group: list, reset_index: bool = True, drop_index: bool = True, 
                                   quantile: float=0.75, iqr_extension: float=1.5, debug: bool = True) -> pd.DataFrame:
    """
    Same as "remove_outliers_iqr_df", but also filter values after divided by the group of columns specified in "group";
    """
    old_len = len(data)
    filtered = []
    for i, g in data.groupby(group, sort=False):
        if debug:
            print(f"filter {i}")
        filtered += [remove_outliers_iqr_df(g, column, reset_index, drop_index, quantile, iqr_extension, debug)]
    new_data = pd.concat(filtered, ignore_index=True)
    if debug and (len(new_data) < old_len):
        print(f"removed a total of {old_len - len(new_data)} outliers")
    return new_data


def compute_speedup(X: pd.DataFrame, col_slow: str, col_fast: str, col_speedup: str) -> None:
    """
    Add a column to a dataframe that represents a speedup,
    and "col_slow", "col_fast" are execution times (e.g. CPU and GPU execution time);
    """
    X[col_speedup] = X[col_slow] / X[col_fast]
    

def correct_speedup_df(data: pd.DataFrame, key: list, baseline_filter_col, baseline_filter_val, speedup_col_name: str="speedup", speedup_col_name_reference: str=None): 
    """
    Divide the speedups in "speedup_col_name" by the geomean of "speedup_col_name_reference", 
    grouping values by the columns in "key" and specifying a baseline column and value to use as reference.
    In most cases, speedup_col_name and speedup_col_name_reference are the same value.
    Useful to ensure that the geomean baseline speedup is 1, and that the other speedups are corrected to reflect that;

    Parameters
    ----------
    data : pd.DataFrame
    key : list
        list of columns on which the grouping is performed, e.g. ["benchmark_name", "implementation"].
    baseline_filter_col : list
        one or more columns used to recognize the baseline, e.g. ["hardware"].
    baseline_filter_val : list
        one or more values in "baseline_filter_col" used to recognize the baseline, e.g. ["cpu"]..
    speedup_col_name : str, optional
        Name of the speedup column to adjust. The default is "speedup".
    speedup_col_name_reference : str, optional
        Name of the reference speedup column, by default it is the same as "speedup_col_name"

    Returns
    -------
    Updated DataFrame

    """
    if not speedup_col_name_reference:
        speedup_col_name_reference = speedup_col_name
    for i, g in data.groupby(key):
        gmean_speedup = gmean(g.loc[g[baseline_filter_col] == baseline_filter_val, speedup_col_name_reference])
        data.loc[g.index, speedup_col_name] /= gmean_speedup    

    
def compute_speedup_df(data: pd.DataFrame, key: list, baseline_filter_col: list, baseline_filter_val: list,  
                    speedup_col_name: str="speedup", time_column: str="exec_time",
                    baseline_col_name: str="baseline_time",
                    correction: bool=True, aggregation=np.median,
                    compute_relative_perf: bool=False):
    """
    Compute speedups on a DataFrame by grouping values

    Parameters
    ----------
    data : pd.DataFrame
    key : list
        list of columns on which the grouping is performed, e.g. ["benchmark_name", "implementation"].
    baseline_filter_col : list
        one or more columns used to recognize the baseline, e.g. ["hardware"].
    baseline_filter_val : list
        one or more values in "baseline_filter_col" used to recognize the baseline, e.g. ["cpu"]..
    speedup_col_name : str, optional
        Name of the new speedup column. The default is "speedup".
    time_column : str, optional
        Name of the execution time column. The default is "exec_time".
    baseline_col_name : str, optional
        Name of the new baseline execution time column. The default is "baseline_time".
    correction : bool, optional
        If True, ensure that the median of the baseline is 1. The default is True.
    aggregation : function, optional
        Function used to aggregate values. The default is np.median.
    compute_relative_perf: bool, optional
        If True, compute relative performance instead of speedup (i.e. 1 / speedup);

    Returns
    -------
    A new DataFrame with the speedups
    """
    
    # Initialize speedup values;
    data[speedup_col_name] = 1
    data[baseline_col_name] = 0
    
    if type(baseline_filter_col) is not list:
        baseline_filter_col = [baseline_filter_col]
    if type(baseline_filter_val) is not list:
        baseline_filter_val = [baseline_filter_val]
        
    assert(len(baseline_filter_col) == len(baseline_filter_val))
        
    grouped_data = data.groupby(key, as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        indices = [group[group[i] == j].index for i, j in zip(baseline_filter_col, baseline_filter_val)]
        reduced_index = reduce(lambda x, y: x.intersection(y), indices)
        mean_baseline = aggregation(data.loc[reduced_index, time_column])
        # Compute the speedup for this group;
        group.loc[:, speedup_col_name] = (group[time_column] / mean_baseline) if compute_relative_perf else (mean_baseline / group[time_column])
        group.loc[:, baseline_col_name] = mean_baseline
        data.loc[group.index, :] = group
    
        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        if correction:
            gmean_speedup = gmean(data.loc[reduced_index, speedup_col_name])
            group.loc[:, speedup_col_name] /= gmean_speedup
            data.loc[group.index, :] = group