import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from segretini_matplottini.plot import roofline
from segretini_matplottini.utils.colors import PALETTE_ORANGE_BASELINE_AND_GREEN_TONES

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401

MARKERS = ["o", "X", "D", "P"]
PALETTE = PALETTE_ORANGE_BASELINE_AND_GREEN_TONES


@save_tmp_plot
def test_default() -> None:
    performance = 0.4 * 10**9
    peak_bandwidth = 140.8 * 10**9
    peak_performance = 666 * 10**9
    operational_intensity = 1 / 12
    roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
    )


@save_tmp_plot
def test_single_roofline() -> None:
    performance = 0.4 * 10**9
    peak_bandwidth = 140.8 * 10**9
    peak_performance = 666 * 10**9
    operational_intensity = 1 / 12
    roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
        add_legend=True,
        legend_labels="CPU",
    )


@save_tmp_plot
def test_stacked_roofline() -> None:
    packet_size = 15
    bandwidth_single_core = 13.2 * 10**9
    max_cores = 64
    clock = 225 * 10**6
    packet_sizes = [packet_size] * 4
    num_cores = [1, 8, 16, 32]
    peak_performance = [p * clock * max_cores for p in packet_sizes]
    peak_bandwidth = [bandwidth_single_core * c for c in num_cores]
    operational_intensity = [p / (512 / 8) for p in packet_sizes]
    exec_times_fpga = [108, 12, 5, 3]
    performance = [20 * 10**7 / (p / 1000) for p in exec_times_fpga]
    num_cores = [1, 8, 16, 32]
    roofline(
        performance,
        operational_intensity,
        peak_performance,
        peak_bandwidth,
        palette=PALETTE,
        markers=MARKERS,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in num_cores],
    )


@save_tmp_plot
def test_double_roofline() -> None:
    packet_size = 15
    bandwidth_single_core = 13.2 * 10**9
    max_cores = 64
    clock = 225 * 10**6
    packet_sizes = [packet_size] * 4
    num_cores = [1, 8, 16, 32]
    peak_performance_1 = [p * clock * max_cores for p in packet_sizes]
    peak_bandwidth_1 = [bandwidth_single_core * c for c in num_cores]
    operational_intensity_1 = [p / (512 / 8) for p in packet_sizes]
    exec_times_fpga = [108, 12, 5, 3]
    performance_1 = [20 * 10**7 / (p / 1000) for p in exec_times_fpga]
    performance_2 = [0.4 * 10**9, 27 * 10**9, 20 * 10**7 / (3 / 1000)]
    operational_intensity_2 = [1 / 12, 1 / 12, 0.23]
    peak_performance_2 = [666 * 10**9, 3100 * 10**9, packet_size * clock * max_cores]
    peak_bandwidth_2 = [140.8 * 10**9, 549 * 10**9, 422.4 * 10**9]
    num_col = 2
    num_row = 1
    fig = plt.figure(figsize=(2.7 * num_col, 2.4 * num_row))
    gs = gridspec.GridSpec(num_row, num_col, top=0.95, bottom=0.2, left=0.1, right=0.92, hspace=0, wspace=0.4)
    # First roofline
    ax = fig.add_subplot(gs[0, 0])
    fig, ax = roofline(
        performance_1,
        operational_intensity_1,
        peak_performance_1,
        peak_bandwidth_1,
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in num_cores],
    )
    # Second roofline
    ax = fig.add_subplot(gs[0, 1])
    fig, ax = roofline(
        performance_2,
        operational_intensity_2,
        peak_performance_2,
        peak_bandwidth_2,
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        ylabel="",
        add_legend=True,
        legend_labels=["CPU", "GPU", "FPGA"],
        add_bandwidth_label=False,
        add_operational_intensity_label=False,
        reset_plot_style=False,
    )
