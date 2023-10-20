
# Changelog

Here you can find a list of the latest updates to `segretini_matplottini`, such as new recipes for plots.

## 2023-10-20

* Added a `get_legend_handles_from_colors` function in `segretini_matplottini.utils` to create the legend handles from a list of colors. Now you can create a personalized legend as follows.

```python
from segretini_matplottini.utils.legend import (
    add_legend_with_dark_shadow,
    get_legend_handles_from_colors,
)
fig, ax = plt.subplots()
palette = ["#48C2A3","#FF6494"]
add_legend_with_dark_shadow(
    handles=get_legend_handles_from_colors(palette),
    labels=["Label A", "Label B"],
    ax=ax,
)
```

* Fixed `timeseries` with `stem` not working when an existing `ax` is passed.
* Added the option to specify `xlim` in `timeseries`.
* Bumped support to Seaborn 0.13.0 and Matplotlib 3.8.0.

## 2023-08-06

🤯 Added an introductory notebook, `notebooks/1_getting_started_with_barplots.ipynb`.
* The notebook provides an introduction to Matplotlib and Seaborn, and guides the reader into creating a beautiful barplot.
* It explains some important concepts behind creating a great visualization, and also a few advanced customization tricks.
This is what you will be able to create in the end.

![Barplot created from the notebook](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/notebooks/1_getting_started_with_barplots/our_amazing_barplot_v3.png)

## 2023-07-26

### Barplots 

Added new `barplot` functions. Creating a barplot is easy, creating a pretty one is not!
* `segretini_matplottini.plot.barplot_for_multiple_categories` wraps Seaborn to plot multiple categories in a single barplot.
* `segretini_matplottini.plot.barplot` plots a single barplot, with the same aesthetic of the other plots.
* `segretini_matplottini.plot.barplots` uses `barplot` to create a grid of barplots, iterating over a list of categories.

![Barplots](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/barplots.png)
![Barplot for multiple categories](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/barplot_for_multiple_categories.png)

### Binary classification plots

When evaluating a binary classification model it is common to evaluate its performance using a variety of metrics and classification thresholds. We added axis-level functions to plot the Confusion Matrix, Precision, Recall, F1, ROC, and Precision-Recall curves, and a figure-level `segretini_matplottini.plot.binary_classification` function to plot all of them at once.

![Binary classification](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/binary_classification.png)

### Many new utilty functions

New utilities in `segretini_matplottini.utils`:
* `add_arrow_to_barplot` adds an up or down arrow to a barplot, to highlight that a higher value or a lower value is better.
* `create_hex_palette` creates a linear palette starting from a starting and ending color, with the requested number of colors. Just a wrapper around `sns.color_palette`, but easier to use.
* `convert_color_to_deficiency` and `convert_colors_to_deficiency` to simulate the impact of different color deficiencies in a color or in a palette of colors.
* `get_labels_for_bars` obtains the labels to add on top of the bars in a barplot, representing the height of each bar.
* `add_labels_to_bars` adds textual labels to a barplot, with options to customize the position and the style of the labels. You can combine it with `get_labels_for_bars` to add labels automatically, or you can manually pass custom labels if to obtain extra control. See `examples/plot_barplot.py` for an example.

## 2022-10-03

Added a new **timeseries plot**, find it it `examples/plot_timeseries.py`. It also supports a sleek `dark background` that you can enable in `segretini_matplottini.utils.reset_style`

![Timeseries](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/timeseries.png)

It can also be used to draw stem plots, by setting `stem=True` in `plot_timeseries`.

![Stem](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/stem.png)

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

