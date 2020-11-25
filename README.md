# segretini-matplottini
A collection of useful matplolib/seaborn tips &amp; tricks collected over years of colorful plot-making

## Repository Structure
* `src/scratchbook.py` contains common commands used to setup plots: creating plots, customizing tick labels, adding custom legends, etc. If you don't remember how to customize some part of your plot, this is a good starting point.
* `src/examples` contains different custom plots. Some are ready-made (like `roofline.py`) and can be used like standard `seaborn` plots. Other examples (like `barplot_2`) are much more complex: look at them if you are trying to replicate some complex feature, like having separate bar groups or adding fancy custom annotations.
* `data` contains files used for plots. For the most part, you can ignore it.
* `plots` is where all the plots are stored. If you find something you like, the code to replicate it is in `src/examples`.

## Update 2020-11-25

Added **Roofline Plot**, find it in `src/examples/roofline.py`.

![Roofline Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/roofline_double.png)

Updated **Ridge Plot**, find it in `src/examples/ridgeplot.py`.

![Ridgeplot Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/ridgeplot.png)

Added **Bar Plot - Example 2**, find it in `src/examples/barplot_2.py`.

![Barplot 2 Example](https://github.com/AlbertoParravicini/segretini-matplottini/blob/master/plots/barplot_2.png)
