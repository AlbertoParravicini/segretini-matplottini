# segretini-matplottini
A collection of useful matplolib/seaborn tips &amp; tricks collected over years of colorful plot-making

## Repository Structure
* `src/scratchbook.py` contains common commands used to setup plots: creating plots, customizing tick labels, adding custom legends, etc. If you don't remember how to customize some part of your plot, this is a good starting point.
* `src/plot_utils.py` contains many useful functions commonly used during plotting, along with nice colors and palettes. Among these functions, some are useful when plotting (for example, to add speedup labels above bars or writing nice labels with exponential notation), while other are useful for data preprocessing (removing outliers, computing speedups).
* `src/examples` contains different custom plots. Some are ready-made (like `roofline.py` and `correlation_scatterplot.py`) and can be used like standard `seaborn` plots. Other examples (like `barplot_2`) are much more complex: look at them if you are trying to replicate some complex feature, like having separate bar groups or adding fancy custom annotations.
* `data` contains files used for plots. For the most part, you can ignore it.
* `plots` is where all the plots are stored. If you find something you like, the code to replicate it is in `src/examples`.

## Tips and Tricks

An ever-growing collection of tips I've found or discovered along the way, together with some nice resources I like a lot

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

## Update 2021-08-28

Update style of **Ridgeplot** to be readable in black & white. Added *large* layout to **Ridgeplot**
![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/ridgeplot_large.png)

Updates to `src/plot_utils.py`: added option to directly provide vertical coordinates to `add_labels`. Added better outlier removal based on interquantile range (the same approach used to find outliers in box-plots)

Added `src/examples/performance_scaling.py`: this plot shows the relative performance increase of processors, memory and interconnection technologies from 1996 to 2021. 
Shamefully copied from [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8) by Amir Gholami.
This plot shows how to use dates for the x-axis, and do fairly complex visualization on log-scale axes (i.e. linear regressions on data with exponential increase).

![Performance Scaling](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/performance_scaling.png)

Temporarily removed `src/examples/barplot_2.py`, it has to be cleaned to be usable again.

## Update 2021-03-22

Updated **Ridgeplot** to have confidence intervals and be more user-friendly (`src/examples/ridgeplot.py`). Added some general tips about choosing colors.

![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/ridgeplot.png)

## Update 2021-03-20

Added **Correlation Scatterplot**, find it in `src/examples/correlation_scatterplot.py`.
Minor updates to `src/plot_utils.py`: new palettes, improved robustness of `get_exp_label`, minor bugfixes.

![Correlation Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/correlation_scatterplot.png)

## Update 2020-11-25

Added **Roofline Plot**, find it in `src/examples/roofline.py`.

![Roofline Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/roofline_double.png)

Updated **Ridge Plot**, find it in `src/examples/ridgeplot.py`.
Added **Bar Plot - Example 2**, find it in `src/examples/barplot_2.py`.
