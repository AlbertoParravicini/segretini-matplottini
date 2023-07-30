from typing import Union

import numpy as np
import seaborn as sns
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb

# Define some colors for later use.
# * Tool to create palettes: https://color.adobe.com/create
# * A guide to make nice palettes: https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/

# It doesn't get any more blue than this;
BLUE_KLEIN = "#002fA7"
# A very strong pink color;
MEGA_PINK = "#FF6494"

# Pastel-tint palette from dark blue to red;
PALETTE_1 = ["#4C9389", "#60b1b0", "#8fd4d4", "#9BD198", "#EFE8AC", "#f9af85", "#f59191"]

# Palette of six green tones, light to dark;
PALETTE_GREEN_TONES_6 = ["#D5FFC7", "#9AE39F", "#73BD8E", "#5EA889", "#4F9E8C", "#3C8585"]
# Palette with orange baseline + green tones;
PALETTE_ORANGE_BASELINE_AND_GREEN_TONES = ["#E8CFB5", "#81C798", "#55A68B", "#358787"]
# Two teal tones suitable as the extremes of a discrete palette;
TWO_TEAL_TONES = ["#B1DEBD", "#4BA3A2"]
# Two similar teal tones suitable as the extremes of a continuous palette;
OTHER_TWO_TEAL_TONES = ["#A5E6C6", "#A2F2B1"]
# Two pink tones suitable as the extremes of a discrete palette;
TWO_PINK_TONES = ["#FFA1C3", "#E8487D"]
# Two similar peach tones suitable as the extremes of a continuous palette;
TWO_PEACH_TONES = ["#FF9868", "#FAB086"]
# A green and a pink colors;
GREEN_AND_PINK_TONES = ("#48C2A3", MEGA_PINK)
# Used for plots with a dark background;
BACKGROUND_BLACK = "#0E1117"


def extend_palette(palette: list[str], new_length: int) -> list[str]:
    """
    Replicate a palette (a list of colors) so that it matches the specified length

    :param palette: A list of colors.
    :param new_length: Desired palette length.
    :return: New extended palette.
    """
    return (palette * int(np.ceil(new_length / len(palette))))[:new_length]


def hex_color_to_grayscale(rgb: Union[str, tuple[int, int, int]]) -> str:
    """
    Convert a color expressed as RGB (either hex or tuple of 3 integers in [0, 255])
    into the corresponding grayscale color, by setting the saturation to 0.

    :param rgb: An input RGB color.
    :return: Output grayscale color, as hex.
    """
    hsv = rgb_to_hsv(to_rgb(rgb))
    hsv[1] = 0  # Set saturation to 0;
    return str(to_hex(hsv_to_rgb(hsv)))


def create_hex_palette(start_hex: str, end_hex: str, number_of_colors: int) -> list[str]:
    """
    Given two colors expressed as hex, create a palette of colors that goes from the first to the second,
    with the specified number of colors in between.

    :param start_hex: First color in the palette, as hex string (e.g. "#FF0000")
    :param end_hex: Second color in the palette, as hex string (e.g. "#00FF00")
    :param number_of_colors: Number of colors in the palette, including the two extremes. Must be >= 2.
    :return: A list of colors, as hex strings.
    """
    assert number_of_colors > 2, f"❌ the number of colors in the palette must be >= 2, not {number_of_colors}"
    return [
        str(c)
        for c in sns.color_palette(f"blend:{start_hex},{end_hex}", n_colors=number_of_colors, as_cmap=False).as_hex()
    ]
