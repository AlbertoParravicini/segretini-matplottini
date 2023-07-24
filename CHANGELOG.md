
# Changelog

Here you can find a list of the latest updates to `matplotlib`, such as new recipes for plots.

## 2022-10-03

Added a new **timeseries plot**, find it it `examples/plot_timeseries.py`. Also added a sleek `dark background` setting in `utils.plot_utils.reset_plot_style`.

![Timeseries](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/timeseries.png)

## 2022-09-25

Revamped the structure of the repository, to make it easier to integrate in other repositories and create pretty plots. Also added a new legend style, and squashed many bugs in `plot_utils.py`

## 2021-08-28

Update style of **Ridgeplot** to be readable in black & white. Added *large* layout to **Ridgeplot**

![Ridgeplot](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/ridgeplot_large.png)

Updates to `plot_utils.py`: added option to directly provide vertical coordinates to `add_labels`. Added better outlier removal based on interquantile range (the same approach used to find outliers in box-plots)

Added `examples/plot_performance_scaling.py`: this plot shows the relative performance increase of processors, memory and interconnection technologies from 1996 to 2021. 
Shamefully copied from [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8) by Amir Gholami.
This plot shows how to use dates for the x-axis, and do fairly complex visualization on log-scale axes (i.e. linear regressions on data with exponential increase).

![Performance Scaling](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/performance_scaling.png)

## 2021-03-22

Updated **Ridgeplot** to have confidence intervals and be more user-friendly (`examples/plot_ridgeplot.py`). Added some general tips about choosing colors.

![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/ridgeplot_compact.png)

## 2021-03-20

Added **Correlation Scatterplot**, find it in `examples/plot_correlation_scatterplot.py`.
Minor updates to `plot_utils.py`: new palettes, improved robustness of `get_exp_label`, minor bugfixes.

![Correlation Scatterplot](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/correlation_scatterplot.png)

## 2020-11-25

Added **Roofline Plot**, find it in `examples/plot_roofline.py`.

![Roofline](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/roofline_double.png)

