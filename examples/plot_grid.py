"""
Create four plots with a uniform style and layout, and assemble them into a grid.
This is done to create a grid of plots to show in the README.md file.

Store and reload the four plots as PNGs, put them as a horizontal grid.
If you were to compose plots in a paper, 
you should create a grid in LaTex and load plots as PDFs, to avoid quality loss;
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from examples.plot_correlation_scatterplot import (
    load_data as load_data_correlation_scatterplot,
)
from examples.plot_ridgeplot import load_data as load_data_ridgeplot
from examples.plot_roofline import load_data_2 as load_data_roofline
from examples.plot_timeseries import load_data_2 as load_data_stem
from segretini_matplottini.plot import (
    correlation_scatterplot,
    ridgeplot,
    roofline,
    timeseries,
)
from segretini_matplottini.utils import assemble_filenames_to_save_plot, save_plot
from segretini_matplottini.utils.colors import (
    BB4,
    BB5,
    G2,
    GREEN_AND_PINK_TONES,
    MEGA_PINK,
)

#########
# Setup #
#########

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

FIGURE_SIZE = (3.5, 3.5)

############
# Plotting #
############


def plot_correlation_scatterplot(output_dir: Path) -> None:
    # Axes limits used in the plot, change them accordingy to your data;
    xlimits = (-0.2, 0.6)
    ylimits = (-0.1, 0.3)
    # Color palette used for plotting;
    palette = ["#3DB88F", GREEN_AND_PINK_TONES[0]]
    data = load_data_correlation_scatterplot()
    correlation_scatterplot(
        data=data,
        x="estimate0",
        y="estimate1",
        hue="significant",
        xlimits=xlimits,
        ylimits=ylimits,
        scatterplot_palette=palette,
        density_color=GREEN_AND_PINK_TONES[1],
        regression_color=GREEN_AND_PINK_TONES[0],
        xlabel="Speedup estimate, method A (%)",
        ylabel="Speedup estimate, method B (%)",
        highlight_negative_area=True,
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="correlation_scatterplot",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )
    plt.close()


def plot_roofline(output_dir: Path) -> None:
    # Create a Roofline model, with multiple lines and custom settings;
    markers = ["o", "X", "D", "P"]
    palette = [MEGA_PINK, G2, BB4, BB5]
    data_dict = load_data_roofline()
    roofline(
        data_dict["performance"],
        data_dict["operational_intensity"],
        data_dict["peak_performance"],
        data_dict["peak_bandwidth"],
        palette=palette,
        markers=markers,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in data_dict["num_cores"]],
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="roofline_stacked",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_stem(output_dir: Path) -> None:
    data = load_data_stem()
    timeseries(
        data,
        xlabel="Time [min]",
        ylabel="Intensity",
        date_format="%H:%M:%S",
        minutes_interval_major_ticks=4,
        minutes_interval_minor_ticks=1,
        dark_background=False,
        draw_style="stem",
        figure_size=FIGURE_SIZE,
        bottom_padding=0.16,
        left_padding=0.14,
        right_padding=0.96,
        top_padding=0.96,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="stem",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


def plot_ridgeplot(output_dir: Path) -> None:
    data = load_data_ridgeplot()
    # Keep only one column worth of data
    unique_plots = list(data["name"].unique())
    num_plots = 6
    data = data[data["name"].isin(unique_plots[:num_plots])]
    ridgeplot(
        data,
        xlabel="Relative Execution Time",
        legend_labels=("Before transformations", "After transformations"),
        plot_confidence_intervals=True,
        xlimits=(0.7, 1.3),
        plot_height=3.5 / num_plots,
        aspect_ratio=num_plots,
        number_of_plot_columns=1,
        bottom_padding=0.2,
        left_padding=0.04,
        right_padding=0.96,
        top_padding=0.99,
    )
    save_plot(
        assemble_filenames_to_save_plot(
            directory=output_dir,
            plot_name="ridgeplot_large",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
            file_format="png",
        ),
        verbose=True,
    )


########
# Main #
########

if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        plot_correlation_scatterplot(output_dir)
        plot_roofline(output_dir)
        plot_stem(output_dir)
        plot_ridgeplot(output_dir)
        # Load the four plots as PNGs, put them as a horizontal grid.
        # This is done to create a grid of plots to show in the README.md file,
        # but if you were to compose plots in a paper,
        # you should create a grid in LaTex and load plots as PDFs, to avoid quality loss;
        fig = plt.figure(figsize=(3.5, 3.5 * 4), dpi=900)
        gs = gridspec.GridSpec(1, 4, hspace=0, wspace=0)
        for i, plot_name in enumerate(["correlation_scatterplot", "roofline_stacked", "stem", "ridgeplot_large"]):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(plt.imread(output_dir / f"{plot_name}.png"))
            ax.axis("off")
        save_plot(
            assemble_filenames_to_save_plot(
                directory=PLOT_DIR,
                plot_name="grid",
                add_timestamp_prefix_to_plot_name=False,
                store_plot_into_timestamp_subfolder=False,
            ),
            verbose=True,
            remove_white_margin=True,
            dpi=900,
        )
