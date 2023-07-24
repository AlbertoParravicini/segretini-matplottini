# segretini-matplottini

A collection of Matplotlib and Seaborn recipes and utilities collected over years of colorful plot-making,
to help researchers create publication-ready plots with ease.

## Installation

Clone the repository and install `segretini-matplottini` with `pip`.

```shell
git clone https://github.com/AlbertoParravicini/segretini-matplottini.git
cd segretini-matplottini
pip install .
```

If you want to hack `segretini-matplottini` itself, you will most likely want to clone it with SSH, install it inside a `conda` environment, in `editable` mode, and with `dev` dependencies.
    
```shell
git clone git@github.com:AlbertoParravicini/segretini-matplottini.git
cd segretini-matplottini
conda create -n segretini_matplottini python=3.9
conda activate segretini_matplottini
pip install -e ".[dev]"
```

# TODO

[x] rename files in `data` to end with `data.csv`
[x] delete scratchbook
[x] add ruff and isort
[x] add pre-commit hooks
[x] linting
[x] reverse gitignore
[] new code structure in readme, with tree
[x] modify plots structure not to have dates
[x] recreate sample plots
[x] create imports for utils inside __init__, to hide private stuff
[x] changelog
[] new barplot code
[] ranz unused colors/palettes, add some new sample colors (e.g. green used in new plots)
[] add util for arrow of metric
[] add util for color palette
[] add image with plot grid inside readme
[] add list of examples to readme, add images for the pretties ones
[] compute_speedup should not mention speedup, and clean API. Maybe ranz correct_speedup, and add test
[] tests
    [] unit tests
    [] e2e tests for examples, check if they don't crash
[] github
    [] protect master branch
    [] add automated tests check
    [] add automated linting check
[x] revisit save_plot. Take as input a path or a list of paths. And have another function to assemble the paths from root dir, plot name (not as format string), folder date (as format string, e.g Y_M_D) and suffix date (_Y_M_D_H_M_S), extensions
[] revisit add_labels: create get_labels_for_barplot and add_labels_to_barplot that takes mandatory labels (and checks that num labels == num bars)
[] create notebook to create good plot from scratch (some barplot)
[] update changelog
[] delete this list

## Repository Structure

* `segretini_matplottini/utils/plot_utils.py` contains many useful functions commonly used during plotting, to add labels above bars or writing nice labels with exponential notation.
* `segretini_matplottini/utils/data_utils.py` contains functions useful for data preprocessing, such as removing outliers, computing and speedups.
* `segretini_matplottini/plots` contains custom plotting functions that can be used like standard Seaborn plots.
* `examples` contains examples that show how to create very complex custom prompts. Look at them if you are trying to replicate some specific feature, like having separate bar groups or adding fancy custom annotations.
* `data` contains files used for plots. For the most part, you can ignore it.
* `plots` is where all the plots are stored. If you find something you like, the code to replicate it is in `examples`.

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

## Development notes

We use `black`, `mypy`, and `ruff` for formatting, type checking, linting, and sorting imports. Each commit must pass their checks. To run checks manually, run the following.

```shell
black . --config pyproject.toml 
mypy . --config mypy.ini
ruff . --fix --config ruff.toml
```
