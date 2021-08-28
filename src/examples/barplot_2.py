# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov 25 16:41:33 2020

# Create a complex barplot with stacked bars, groups, and many many labels; 

# @author: albyr
# """

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt 
# import matplotlib.gridspec as gridspec
# import scipy.stats as st
# from matplotlib.patches import Patch
# import os
# from scipy.stats.mstats import gmean
# import matplotlib.lines as lines
# import matplotlib.ticker as ticker

# import sys
# sys.path.append("..")
# from plot_utils import *


# RES_FOLDER = "../../data/ridgeplot_data"

# KERNELS =  ["axpy", "dot_product", "convolution", "mmul", "autocov", "hotspot", "hotspot3d",
#             "backprop", "backprop2", "bfs", "pr", "nested", "gaussian",
#             "histogram", "lud", "needle"]

# PALETTE = [COLORS["peach2"], COLORS["g1"]]
# PALETTE_B = [COLORS["b3"], COLORS["b3"]]
# HATCHES = ["/" * 4, "\\" * 4]


# def barplot(res):
     
#     # We want to add empty spots in the plot, to separate groups of bars. 
#     # To do so, simply insert fake entries in the order of bars that we want to plot;
#     group_end_1 = 1
#     group_end_2 = 10
#     group_ends = [group_end_1, group_end_2]
#     order = list(res["kernel"].unique())
#     # Add "GMEAN" as first;
#     order.remove("GMEAN")
#     order = ["GMEAN"] + order
#     # Add empty spots to separate groups;
#     order = order[0:group_end_1] + ["FAKE1"] + order[group_end_1:group_end_2] + ["FAKE2"] + order[group_end_2:]
    
#     num_benchmarks = len(order)
    
#     sns.set_style("white", {"ytick.left": True})
#     plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
#     plt.rcParams['hatch.linewidth'] = 0.3
#     plt.rcParams['axes.labelpad'] = 5 
    
#     fig = plt.figure(figsize=(7.5, 2.5))
#     gs = gridspec.GridSpec(1, 1)
#     plt.subplots_adjust(top=0.85,
#                         bottom=0.27,
#                         left=0.08,
#                         right=0.98,
#                         hspace=1.0,
#                         wspace=0.8)  
#     ax = fig.add_subplot(gs[0, 0])
#     ax = sns.barplot(x="kernel", y="slowdown", hue="modified", data=res, ci=99, order=order,
#                       palette=PALETTE_B, capsize=.05, errwidth=0.8, ax=ax, edgecolor="#2f2f2f", estimator=gmean, saturation=0.9, zorder=2)
#     ax.legend_.remove()  # Hack to remove legend;

#     # Set of bars that are "fake" and not shown;
#     FORBIDDEN_SET = []
#     for i, g in enumerate(group_ends):
#         FORBIDDEN_SET += [i + g, i + g + num_benchmarks]

#     # Hide tick labels of fake bars;
#     xticks = ax.xaxis.get_major_ticks()
#     for i, f in enumerate(group_ends):
#         xticks[f + i].set_visible(False)

#     # Add label below each bar;
#     labels = ["A"] * num_benchmarks + ["B"] * num_benchmarks
#     for i, p in enumerate(ax.patches):
#         if i not in FORBIDDEN_SET:
#             ax.text(p.get_x() + p.get_width()/2., -0.13, labels[i], fontsize=8, color="#2f2f2f", ha='center')
#     # Add a second level of tick, highlighted with a T shape;
#     ax.tick_params(axis='x', which='major', pad=17, labelsize=8)
#     group_length = len(res["modified"].unique())
#     for i in range(len(ax.patches) // group_length):
#         if i not in FORBIDDEN_SET:
#             y_min = -0.2
#             y_max = -0.25
#             x_middle = (ax.patches[i].get_x() + ax.patches[i + (len(ax.patches) // 2)].get_x() + p.get_width()) / 2
#             ax.plot([ax.patches[i].get_x(), ax.patches[i + (len(ax.patches) // 2)].get_x() + p.get_width()], [y_min, y_min], clip_on=False, color="#2f2f2f", linewidth=1)
#             ax.plot([x_middle, x_middle], [y_min, y_max], clip_on=False, color="#2f2f2f", linewidth=1)
        
#     # Speedup labels;
#     offsets = []
#     for k, g in res.groupby(["kernel", "modified"], sort=False):
#         offsets += [get_upper_ci_size(g["slowdown"], ci=0.99)]
#     offsets = [max(o, 0.07) for o in offsets]
#     add_labels(ax, vertical_offsets=offsets, rotation=0, format_str="{:.2f}", fontsize=8, skip_zero=False, skip_value=1)
    
#     # Add other bars above;
#     ax = sns.barplot(x="kernel", y="slowdown_k", hue="modified", data=res, ci=None, order=order,
#               palette=PALETTE, capsize=.05, errwidth=0.8, ax=ax, edgecolor="#2f2f2f", estimator=gmean, saturation=0.9, zorder=3)
    
#     # hatches2 = [hatches[0]] * (len(ax.patches) // 4) + [hatches[1]] * (len(ax.patches) // 4)
#     for j, bar in enumerate([p for p in ax.patches if not pd.isna(p)][2*num_benchmarks:]):
#         bar.set_hatch(HATCHES[j // num_benchmarks])
    
#     ax.set_ylabel("Relative Exec. Time", fontsize=9)
#     ax.set_xlabel("")
#     ax.set_ylim((0.0, 1.5))
#     ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))
#     ax.yaxis.set_major_locator(plt.LinearLocator(7))
#     ax.tick_params(axis='y', which='major', labelsize=8)
#     ax.grid(True, axis="y")
    
#     # Add legend;
#     ax.legend_.remove()  # Remove the existing legend again;
#     labels = ["Baseline", "Modified", "Overall Time"]
#     custom_lines = [Patch(facecolor=PALETTE[0], hatch=HATCHES[0], edgecolor="#2f2f2f", label=labels[0]),
#                     Patch(facecolor=PALETTE[1], hatch=HATCHES[1], edgecolor="#2f2f2f", label=labels[1]),
#                     Patch(facecolor=PALETTE_B[0], edgecolor="#2f2f2f", label=labels[2])] 
                    
#     leg = fig.legend(custom_lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0), fontsize=7, ncol=3, handletextpad=0.5, columnspacing=0.4)
#     leg.set_title(None)
#     leg._legend_box.align = "left"
#     leg.get_frame().set_facecolor('white')
    
#     ax.axhline(y=1, color="#2f2f2f", linestyle="--", zorder=1, linewidth=1, alpha=0.5)
#     ax.axvline(x=(ax.patches[1].get_x() + ax.patches[1].get_width()), color="#2f2f2f", linestyle=":", zorder=1, linewidth=1, alpha=0.9)
#     ax.axvline(x=(ax.patches[11].get_x() + ax.patches[11].get_width()), color="#2f2f2f", linestyle=":", zorder=1, linewidth=1, alpha=0.9)
    
#     # Add grouping bars above the plot;
#     y_min = 1.57
#     y_max = 1.62
#     x_start = ax.patches[group_end_1 + 1].get_x()
#     x_end = ax.patches[group_end_2 + (len(ax.patches) // 4)].get_x() + ax.patches[group_end_2 + (len(ax.patches) // 4)].get_width()
#     x_middle = (x_start + x_end) / 2
#     ax.plot([x_start, x_end], [y_min, y_min], clip_on=False, color="#2f2f2f", linewidth=1)
#     ax.plot([x_middle, x_middle], [y_min, y_max], clip_on=False, color="#2f2f2f", linewidth=1)
#     ax.text(x_middle, y_max + 0.05, "Group A", fontsize=8, ha="center")
    
#     x_start = ax.patches[group_end_2 + 2].get_x()
#     x_end = ax.patches[-1].get_x() + ax.patches[-1].get_width()
#     x_middle = (x_start + x_end) / 2
#     ax.plot([x_start, x_end], [y_min, y_min], clip_on=False, color="#2f2f2f", linewidth=1)
#     ax.plot([x_middle, x_middle], [y_min, y_max], clip_on=False, color="#2f2f2f", linewidth=1)
#     ax.text(x_middle, y_max + 0.05, "Group B", fontsize=8, ha="center")
    
#     return fig, ax


# def load_exec_data(path, kernel):
#     res = pd.read_csv(path, sep=", ") 
#     res = res.dropna(subset=["exec_time_u_k_us"])
#     res["kernel"] = kernel
#     res = remove_outliers_df_grouped(res, "exec_time_u_k_us", ["kernel", "opt_level", "simplify"])
#     res = remove_outliers_df_grouped(res, "exec_time_m_k_us", ["kernel", "opt_level", "simplify"])
#     res = remove_outliers_df_grouped(res, "exec_time_u_us", ["kernel", "opt_level", "simplify"])
#     res = remove_outliers_df_grouped(res, "exec_time_m_us", ["kernel", "opt_level", "simplify"])
    
#     res1 = pd.melt(res, id_vars=["kernel", "iteration", "opt_level", "simplify"],
#                   value_vars=["exec_time_u_us", "exec_time_m_us"], var_name="modified", value_name="exec_time_us")
#     res2 = pd.melt(res, id_vars=["kernel", "iteration", "opt_level", "simplify"],
#               value_vars=["exec_time_u_k_us", "exec_time_m_k_us"], var_name="modified", value_name="exec_time_k_us")
#     res1["modified"] = res1["modified"] == "exec_time_m_us"
#     res1["exec_time_k_us"] = res2["exec_time_k_us"]
           
#     compute_speedup_df(res1, ["kernel", "opt_level", "simplify"], ["modified"], [False], time_column="exec_time_us", baseline_col_name="baseline_us")
#     res1["slowdown"] = 1 / res1["speedup"]
    
#     # Relative execution time of the kernel (unmodified and modified) w.r.t total exec time of unmodified;
#     res1["speedup_k"] = 1
#     res1["baseline_k_us"] = 0
    
#     grouped_data = res1.groupby(["kernel", "opt_level", "simplify"], as_index=False)
#     for group_key, group in grouped_data:
#         # Compute the median baseline computation time;
#         median_baseline = np.median(group.loc[group["modified"] == False, "exec_time_us"])
#         # Compute the speedup for this group;
#         group.loc[:, "speedup_k"] = median_baseline / group["exec_time_k_us"]
#         group.loc[:, "baseline_k_us"] = median_baseline
#         res1.loc[group.index, :] = group
    
#     res1["slowdown_k"] = 1 / res1["speedup_k"]
#     return res1


# if __name__ == "__main__":
    
#     ##################################
#     # Load data ######################
#     ##################################
    
#     # We load execution time results of different benchmarks, grouped in "baseline" and "modified" code ("modified" column False or True).
#     # For each benchmark, we also have the "pure execution time" (_k columns), and the time comprehensive of some setup overhead.
#     # We compute the relative execution time (i.e. slowdown) of "modified" benchmarks w.r.t. the "baseline", 
#     # and also the proprtion of "pure execution time" w.r.t. time with overheads.    
#     res_list = []
#     for i, k in enumerate(KERNELS):
#         for f in os.listdir(RES_FOLDER):
#             if os.path.splitext(f)[0] == k:
#                 res_list += [load_exec_data(os.path.join(RES_FOLDER, f), f"B{i}")]
#     res = pd.concat(res_list, ignore_index=True).reset_index(drop=True)
#     # Only simplified accesses;
#     res = res[res["simplify"] == "simplify_accesses"]
#     # Only O2 level;
#     res = res[res["opt_level"] == "O2"]
    
#     res_grouped = res.groupby(["kernel", "opt_level", "modified"]).agg(gmean).reset_index()
#     # Guarantee that the geomean execution time of the baseline total time is 1, and that all relative execution times are corrected w.r.t. that value;
#     correct_speedup_df(res_grouped, ["opt_level"], "modified", False, "speedup")
#     correct_speedup_df(res_grouped, ["opt_level"], "modified", False, "speedup_k", speedup_col_name_reference="speedup")
#     correct_speedup_df(res_grouped, ["opt_level"], "modified", False, "slowdown")
#     correct_speedup_df(res_grouped, ["opt_level"], "modified", False, "slowdown_k", speedup_col_name_reference="slowdown")
#     res_grouped_2 = res_grouped.groupby(["opt_level", "modified"]).agg(gmean)
#     # Add geomean. We do that by adding a "fake" benchmark with name GMEAN that contains the average speed of 
#     #  all the other benchmarks (so that we can aggregate them and obtain the variance);
#     res_geomean = res_grouped.copy()
#     res_geomean["kernel"] = "GMEAN"
#     res = pd.concat([res_geomean, res], ignore_index=True).reset_index(drop=True)
        
#     # res = add_fake_row(res, "kernel")
    
#     ##################################
#     # Plotting #######################
#     ##################################  
    
#     fig, ax = barplot(res)
#     save_plot("../../plots", "barplot_2.{}")  
    