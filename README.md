# segretini-matplottini

A collection of Matplotlib and Seaborn recipes and utilities collected over years of colorful plot-making,
to help researchers create publication-ready plots with ease.

👇 For example, a `correlation_scatterplot`, a stacked `roofline` plot, a `timeseries` plot with `stems`, and a `ridgeplot`. Find other examples below!

![Grid of example plots](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/grid.png)

## 🚂 Installation

Clone the repository and install `segretini-matplottini` with `pip`.

```shell
git clone https://github.com/AlbertoParravicini/segretini-matplottini.git
cd segretini-matplottini
pip install .
```

If you want to play with notebooks, also install `jupyter`.

```shell
pip install ".[notebook]"
```

## ✍️ Repository Structure

These are the most important folders of `segretini-matplottini`, so you won't get lost when exploring the code.

```
segretini-matplottini (this folder)
    ├── segretini_matplottini
    │   ├── plot              -> Ready-made plotting functions (e.g. ridgeplots, and Roofline plots)
    │   └── utils             -> Other setup files (constants, metadata, etc.)
    │       ├── constants.py  -> Default values for shared settings (font size, DPI, etc.)
    │       ├── colors.py     -> List of predefined and pretty colors, plus color-related utilities
    │       ├── data.py       -> Utilities to preprocess datasets (e.g. outlier removal)
    │       ├── legend.py     -> Legend-related utilities (e.g. custom legend styles)
    │       └── plot.py       -> General utilities for plotting (e.g. adding labels, saving plots)
    ├── data         -> Sample data used in example plots
    ├── examples     -> Recipes to create fancy plots
    ├── plots        -> Plots generated by examples. If you find something cool, check the code in examples
    ├── tests        -> Unit tests and end-to-end tests for plotting functions
    ├── CHANGELOG.md -> List of updates to the codebase. Check there to see what's new
    ├── README.md   -> This file!
    └── (...)       -> Configuration files for linters, and other setup files.
```

## 🌞 Getting started

The best way to get started is to check out the [`plots`](plots/) folder, to find plots generated with `segretini-matplottini`.
If you find a plot you like, you can find the code to generate it in the [`examples`](examples/) folder.

### Some plots available in `segretini-matplottini`

This is a non-inclusive list of custom plots that are available out-of-the-box in `segretini-matplottini`.
* `correlation_scatterplot` visualizes the relation between two variables, combining a scatterplot, a 2D density plot, and a linear regression with confidence intervals. Learn more with [this](examples/plot_correlation_scatterplot.py) example. 
* `ridgeplot` shows the distribution of two variables, grouped by the specified factor. For example, one can visualize the latency of two implementations of the same algorithm, across multiple runs of different datasets. Learn more with [this](examples/plot_ridgeplot.py) example. 
* `roofline` plots the [Roofline model](https://en.wikipedia.org/wiki/Roofline_model) for the input operational intensity and performance values. Learn more with [this](examples/plot_roofline.py) example. 
* `binary_classfication` summarizes the performance of a binary classifier, plotting curves such as ROC, Precision-Recall, and F1 score for different classfication thresholds. Learn more with [this](examples/plot_binary_classification.py) example.

### Some utilities available in `segretini-matplottini`

The astute reader might say "Hey, I don't need any of those plots, why should I care about `segretini-matplottini`?".
There's a lot more than plotting functions! The `utils` can be applied to any Matplotlib plot, to simplify your life when it comes to create complex visualizations.
* [`data`](segretini_matplottini/utils/data.py) contains functions to preprocess your experiment results by [removing outliers](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/data.py#L158) and computing the [relative performance](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/data.py#L242) from absolute performance numbers.
* [`colors`](segretini_matplottini/utils/colors.py) provides utilities to convert your palettes to [grayscale](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/colors.py#L48) to check how your plot will look when printed in black and white or when seen by a color deficient person, and to simplify the [creation of palettes](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/colors.py#L61) given the start and end colors. It also has plenty of beautiful colors to choose from, validated for black and white printing and color blindness.
* [`plot`](segretini_matplottini/utils/plot.py) is the source for general plotting utilities, from [computing](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/plot.py#L323) and [adding](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/plot.py#L390) labels to barplots to [saving plots](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/segretini_matplottini/utils/plot.py#L461) with a standardized structure, so they won't get lost or overwritten by accident.

In the examples below, a `binary_classification` plot with a few of the available sub-plots turned on, and a `barplot` that takes advantage of the `utils` to compute relative performance and add labels to bars.

![Grid of example plots, binary classification and barplot](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/grid_2.png)

## 👨‍🔬 Development notes

If you want to hack `segretini-matplottini` itself, you will most likely want to clone it with SSH, install it inside a `conda` environment, in `editable` mode, and with `dev` dependencies. Also install the `pre-commit` hooks to obtain linting and quality checks when creating a commit.
    
```shell
git clone git@github.com:AlbertoParravicini/segretini-matplottini.git
cd segretini-matplottini
conda create -n segretini_matplottini python=3.9
conda activate segretini_matplottini
pip install -e ".[dev,notebook]"
pre-commit install
```

We use `black`, `mypy`, and `ruff` for formatting, type checking, linting, and sorting imports. Each commit must pass these checks. To run checks manually, run the following commands. Checks also run when creating a commit. You won't be able to commit if a check fails.

```shell
black . --config pyproject.toml 
mypy . --config mypy.ini
ruff . --fix --config ruff.toml
```

## 💡 Tips and Tricks

An ever-growing collection of tips I've found or discovered along the way, together with some nice resources I like a lot.

### 📚 Resources 

* [Subtleties of Color](https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/): a six-parts guide on how to pick nice colors and create great palettes, from the Earth Observatory NASA blog.
* [Adobe Color](https://color.adobe.com/create/color-wheel): a free web tool to create palettes of different types (shades, complementary, etc.). It also has accessibility tools to test for color blindness safety.

### 🎨 Colors

Picking the right colors is hard! I found the following tips to be very helpful.

* **Not everyone sees colors in the same way**: most reviewers will look at your papers after printing them in black & white. Always check for that! You can use `segretini_matplottini.utils.convert_color_to_grayscale` or doing a print preview after saving your plot as PDF.
If colors are too similar, try adjusting the **L** (lightness) in the **HSL** representation, or the **B** (brightness) in the **HSB** representation. 
Also, a lot of people are color blind, and there are many types of color blindness! It's always better to double check.
To do so, you can use `segretini_matplottini.utils.convert_color_to_deficiency`.

* **Hidden color biases**: people tend to associate implicit meanings to colors. To simplify the matter a lot, green is usually associated to positive things, while red is bad. 
If your plot is not explicitely comparing classes (for example, you want to show the speed of your algorithm on different datasets), just go for a neutral/positive palette, using colors such as gree, blue, and light pink.

* **Add redundant information**: if you are plotting many different classes, and use one color per class, it can be difficult to distinguish among them. Instead, add some kind of redundant information.
In scatterplots and lineplots you can use different markers (circles, diamonds, etc.), while in barplots you can use different hatches (//// or \\\\) or add labels to each class.
