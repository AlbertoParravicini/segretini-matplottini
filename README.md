# segretini-matplottini
A collection of useful Matplolib and Seaborn tips &amp; tricks collected over years of colorful plot-making.

## Installation

Simply run the following.
```
git clone git@github.com:AlbertoParravicini/segretini-matplottini.git
pip install segretini-matplottini
```

## Repository Structure

* `segretini_matplottini/utils/plot_utils.py` contains many useful functions commonly used during plotting, to add labels above bars or writing nice labels with exponential notation.
* `segretini_matplottini/utils/data_utils.py` contains functions useful for data preprocessing, such as removing outliers, computing and speedups.
* `segretini_matplottini/plots` contains custom plotting functions that can be used like standard Seaborn plots.
* `notebooks` contains examples that show how to create very complex custom prompts. Look at them if you are trying to replicate some specific feature, like having separate bar groups or adding fancy custom annotations.
* `notebooks/plot_scratchbook.py` contains common commands used to setup plots: creating plots, customizing tick labels, adding custom legends, etc. If you don't remember how to customize some part of your plot, this is a good starting point.
* `data` contains files used for plots. For the most part, you can ignore it.
* `plots` is where all the plots are stored. If you find something you like, the code to replicate it is in `notebooks`.

## Tips and Tricks

An ever-growing collection of tips I've found or discovered along the way, together with some nice resources I like a lot.

### Resources

* [Subtleties of Color](https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/): a six-parts guide on how to pick nice colors and create great palettes, from the Earth Observatory NASA blog.
* [Adobe Color](https://color.adobe.com/create/color-wheel): a free web tool to create palettes of different types (shades, complementary, etc.). It also has accessibility tools to test for color blindness safety.

### Colors

Picking the right colors is hard! I found the following tips to be very helpful.

* **Not everyone sees colors in the same way**: most reviewers will look at your papers after printing them in black & white. Always check for that! You can do it with `matplotlib` (check out `hex_color_to_grayscale` in `src/plot_utils.py`) or doing a print preview after saving your plot as PDF.
If colors are too similar, try adjusting the **L** (lightness) in the **HSL** representation, or the **B** (brightness) in the **HSB** representation. 
Also, a lot of people are color blind, and there are many types of color blindness! It's always better to double check.

* **Hidden color biases**: people tend to associate implicit meanings to colors. To simplify the matter a lot, green is usually associated to positive things, while red is bad. 
If your plot is not explicitely comparing classes (for example, you want to show the speed of your algorithm on different datasets), just go for a neutral/positive palette, using colors such as gree, blue, and light pink.

* **Add redundant information**: if you are plotting many different classes, and use one color per class, it can be difficult to distinguish among them. Instead, add some kind of redundant information.
In scatterplots and lineplots you can use different markers (circles, diamonds, etc.), while in barplots you can use different hatches (//// or \\\\) or add labels to each class.

## Update 2022-09-25

I'm revamping the structure of the repository, to make it easier to integrate in other repositories and create pretty plots.
* I also added a new legend style, and squashed many bugs in `plot_utils.py`

## Update 2021-08-28

Update style of **Ridgeplot** to be readable in black & white. Added *large* layout to **Ridgeplot**

![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/2022-09-25/ridgeplot_large.png)

Updates to `plot_utils.py`: added option to directly provide vertical coordinates to `add_labels`. Added better outlier removal based on interquantile range (the same approach used to find outliers in box-plots)

Added `notebooks/plot_performance_scaling.py`: this plot shows the relative performance increase of processors, memory and interconnection technologies from 1996 to 2021. 
Shamefully copied from [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8) by Amir Gholami.
This plot shows how to use dates for the x-axis, and do fairly complex visualization on log-scale axes (i.e. linear regressions on data with exponential increase).

![Performance Scaling](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/2022-09-25/performance_scaling.png)

## Update 2021-03-22

Updated **Ridgeplot** to have confidence intervals and be more user-friendly (`notebooks/plot_ridgeplot.py`). Added some general tips about choosing colors.

<!-- ![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/2022-09-25/ridgeplot.png) -->

## Update 2021-03-20

Added **Correlation Scatterplot**, find it in `notebooks/plot_correlation_scatterplot.py`.
Minor updates to `plot_utils.py`: new palettes, improved robustness of `get_exp_label`, minor bugfixes.

![Correlation Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/2022-09-25/correlation_scatterplot.png)

## Update 2020-11-25

Added **Roofline Plot**, find it in `notebooks/plot_roofline.py`.

![Roofline Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/2022-09-25/roofline_double.png)

