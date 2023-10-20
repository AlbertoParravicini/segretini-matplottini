from typing import Literal, Union

import numpy as np
import seaborn as sns
from jaxtyping import Float
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

PROTANOPIA = np.array([[0.56667, 0.43333, 0], [0.55833, 0.44167, 0], [0, 0.24167, 0.75833]])
DEUTERANOMALY = np.array([[0.8, 0.2, 0.0], [0.258, 0.742, 0.0], [0.0, 0.142, 0.858]])


def extend_palette(palette: list[str], new_length: int) -> list[str]:
    """
    Replicate a palette (a list of colors) so that it matches the specified length

    :param palette: A list of colors.
    :param new_length: Desired palette length.
    :return: New extended palette.
    """
    return (palette * int(np.ceil(new_length / len(palette))))[:new_length]


def convert_color_to_grayscale(
    color: Union[str, tuple[int, int, int]], color_space: Literal["hsv", "hls"] = "hsv"
) -> str:
    """
    Convert a color expressed as RGB (either hex or tuple of 3 integers in [0, 255])
    into the corresponding grayscale color, by setting the saturation to 0.

    :param color: An input RGB color.
    :param color_space: The color space to use for the conversion.
        The saturation is set to 0 in all cases, but the resulting grayscale color is different.
    :return: Output grayscale color, as hex.
    """
    if color_space == "hls":
        return str(to_hex(sns.desaturate(color, prop=0)))
    elif color_space == "hsv":
        hsv_color = rgb_to_hsv(to_rgb(color))
        hsv_color[1] = 0
        return str(to_hex(hsv_to_rgb(hsv_color)))  # type: ignore


def convert_colors_to_grayscale(
    colors: Union[list[str], list[tuple[int, int, int]]], color_space: Literal["hsv", "hls"] = "hsv"
) -> list[str]:
    """
    Convert a list of colors into the corresponding grayscale palette,
    by setting the saturation to 0.

    :param palette: A list of colors, as hex strings or RGB tuples.
    :param color_space: The color space to use for the conversion.
        The saturation is set to 0 in all cases, but the resulting grayscale color is different.
    :return: Output grayscale palette, as hex strings.
    """
    return [convert_color_to_grayscale(c, color_space=color_space) for c in colors]


def create_hex_palette(start_hex: str, end_hex: str, number_of_colors: int) -> list[str]:
    """
    Given two colors expressed as hex, create a palette of colors that goes from the first to the second,
    with the specified number of colors in between. Interpolation is done in HSB space.

    :param start_hex: First color in the palette, as hex string (e.g. "#FF0000")
    :param end_hex: Second color in the palette, as hex string (e.g. "#00FF00")
    :param number_of_colors: Number of colors in the palette, including the two extremes. Must be >= 2.
    :return: A list of colors, as hex strings.
    """
    assert number_of_colors >= 2, f"❌ the number of colors in the palette must be >= 2, not {number_of_colors}"
    return [
        str(c)
        for c in sns.color_palette(f"blend:{start_hex},{end_hex}", n_colors=number_of_colors, as_cmap=False).as_hex()
    ]


# 3x3 matrices to simulate color vision deficiencies, for different strenghts.
# Reference: https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
# @article{Machado2009,
#    author    = {Gustavo M. Machado and Manuel M. Oliveira and Leandro A. F. Fernandes},
#    title     = {A Physiologically-based Model for Simulation of Color Vision Deficiency},
#    journal   = {IEEE Transactions on Visualization and Computer Graphics},
#    volume    = {15},
#    number    = {6},
#    month     = {November/December},
#    year      = {2009},
#    pages     = {1291-1298} ,
#    publisher = {IEEE Computer Society}
# }
# These are also the same matrices used by Adobe Color;
COLOR_VISION_DEFICIENCY_MATRICES: dict[
    Literal["protanomaly", "deuteranomaly", "tritanomaly"], dict[float, Float[np.ndarray, "3 3"]]
] = {
    "protanomaly": {
        0.0: np.array(
            [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]]
        ),
        0.1: np.array(
            [
                [0.856167, 0.182038, -0.038205],
                [0.029342, 0.955115, 0.015544],
                [-0.002880, -0.001563, 1.004443],
            ]
        ),
        0.2: np.array(
            [
                [0.734766, 0.334872, -0.069637],
                [0.051840, 0.919198, 0.028963],
                [-0.004928, -0.004209, 1.009137],
            ]
        ),
        0.3: np.array(
            [
                [0.630323, 0.465641, -0.095964],
                [0.069181, 0.890046, 0.040773],
                [-0.006308, -0.007724, 1.014032],
            ]
        ),
        0.4: np.array(
            [
                [0.539009, 0.579343, -0.118352],
                [0.082546, 0.866121, 0.051332],
                [-0.007136, -0.011959, 1.019095],
            ]
        ),
        0.5: np.array(
            [
                [0.458064, 0.679578, -0.137642],
                [0.092785, 0.846313, 0.060902],
                [-0.007494, -0.016807, 1.024301],
            ]
        ),
        0.6: np.array(
            [
                [0.385450, 0.769005, -0.154455],
                [0.100526, 0.829802, 0.069673],
                [-0.007442, -0.022190, 1.029632],
            ]
        ),
        0.7: np.array(
            [
                [0.319627, 0.849633, -0.169261],
                [0.106241, 0.815969, 0.077790],
                [-0.007025, -0.028051, 1.035076],
            ]
        ),
        0.8: np.array(
            [
                [0.259411, 0.923008, -0.182420],
                [0.110296, 0.804340, 0.085364],
                [-0.006276, -0.034346, 1.040622],
            ]
        ),
        0.9: np.array(
            [
                [0.203876, 0.990338, -0.194214],
                [0.112975, 0.794542, 0.092483],
                [-0.005222, -0.041043, 1.046265],
            ]
        ),
        1.0: np.array(
            [
                [0.152286, 1.052583, -0.204868],
                [0.114503, 0.786281, 0.099216],
                [-0.003882, -0.048116, 1.051998],
            ]
        ),
    },
    "deuteranomaly": {
        0.0: np.array(
            [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]]
        ),
        0.1: np.array(
            [
                [0.866435, 0.177704, -0.044139],
                [0.049567, 0.939063, 0.011370],
                [-0.003453, 0.007233, 0.996220],
            ]
        ),
        0.2: np.array(
            [
                [0.760729, 0.319078, -0.079807],
                [0.090568, 0.889315, 0.020117],
                [-0.006027, 0.013325, 0.992702],
            ]
        ),
        0.3: np.array(
            [
                [0.675425, 0.433850, -0.109275],
                [0.125303, 0.847755, 0.026942],
                [-0.007950, 0.018572, 0.989378],
            ]
        ),
        0.4: np.array(
            [
                [0.605511, 0.528560, -0.134071],
                [0.155318, 0.812366, 0.032316],
                [-0.009376, 0.023176, 0.986200],
            ]
        ),
        0.5: np.array(
            [
                [0.547494, 0.607765, -0.155259],
                [0.181692, 0.781742, 0.036566],
                [-0.010410, 0.027275, 0.983136],
            ]
        ),
        0.6: np.array(
            [
                [0.498864, 0.674741, -0.173604],
                [0.205199, 0.754872, 0.039929],
                [-0.011131, 0.030969, 0.980162],
            ]
        ),
        0.7: np.array(
            [
                [0.457771, 0.731899, -0.189670],
                [0.226409, 0.731012, 0.042579],
                [-0.011595, 0.034333, 0.977261],
            ]
        ),
        0.8: np.array(
            [
                [0.422823, 0.781057, -0.203881],
                [0.245752, 0.709602, 0.044646],
                [-0.011843, 0.037423, 0.974421],
            ]
        ),
        0.9: np.array(
            [
                [0.392952, 0.823610, -0.216562],
                [0.263559, 0.690210, 0.046232],
                [-0.011910, 0.040281, 0.971630],
            ]
        ),
        1.0: np.array(
            [
                [0.367322, 0.860646, -0.227968],
                [0.280085, 0.672501, 0.047413],
                [-0.011820, 0.042940, 0.968881],
            ]
        ),
    },
    "tritanomaly": {
        0.0: np.array(
            [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]]
        ),
        0.1: np.array(
            [
                [0.926670, 0.092514, -0.019184],
                [0.021191, 0.964503, 0.014306],
                [0.008437, 0.054813, 0.936750],
            ]
        ),
        0.2: np.array(
            [
                [0.895720, 0.133330, -0.029050],
                [0.029997, 0.945400, 0.024603],
                [0.013027, 0.104707, 0.882266],
            ]
        ),
        0.3: np.array(
            [
                [0.905871, 0.127791, -0.033662],
                [0.026856, 0.941251, 0.031893],
                [0.013410, 0.148296, 0.838294],
            ]
        ),
        0.4: np.array(
            [
                [0.948035, 0.089490, -0.037526],
                [0.014364, 0.946792, 0.038844],
                [0.010853, 0.193991, 0.795156],
            ]
        ),
        0.5: np.array(
            [
                [1.017277, 0.027029, -0.044306],
                [-0.006113, 0.958479, 0.047634],
                [0.006379, 0.248708, 0.744913],
            ]
        ),
        0.6: np.array(
            [
                [1.104996, -0.046633, -0.058363],
                [-0.032137, 0.971635, 0.060503],
                [0.001336, 0.317922, 0.680742],
            ]
        ),
        0.7: np.array(
            [
                [1.193214, -0.109812, -0.083402],
                [-0.058496, 0.979410, 0.079086],
                [-0.002346, 0.403492, 0.598854],
            ]
        ),
        0.8: np.array(
            [
                [1.257728, -0.139648, -0.118081],
                [-0.078003, 0.975409, 0.102594],
                [-0.003316, 0.501214, 0.502102],
            ]
        ),
        0.9: np.array(
            [
                [1.278864, -0.125333, -0.153531],
                [-0.084748, 0.957674, 0.127074],
                [-0.000989, 0.601151, 0.399838],
            ]
        ),
        1.0: np.array(
            [
                [1.255528, -0.076749, -0.178779],
                [-0.078411, 0.930809, 0.147602],
                [0.004733, 0.691367, 0.303900],
            ]
        ),
    },
}


def _find_closest_color_deficiency_matrices(
    deficiency: Literal["protanomaly", "deuteranomaly", "tritanomaly"], strength: float
) -> tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3 3"]]:
    # If strength is 0.76, we obtain the matrices for 0.7 and 0.8.
    # Find the two closest strengths;
    k1, k2 = sorted(COLOR_VISION_DEFICIENCY_MATRICES[deficiency].keys(), key=lambda x: abs(x - strength))[:2]
    # Ensure that the matrix with smallest strength is the first one;
    k1, k2 = sorted([k1, k2])
    return COLOR_VISION_DEFICIENCY_MATRICES[deficiency][k1], COLOR_VISION_DEFICIENCY_MATRICES[deficiency][k2]


def _interpolate_color_deficiency_matrices(
    deficiency: Literal["protanomaly", "deuteranomaly", "tritanomaly"], strength: float
) -> Float[np.ndarray, "3 3"]:
    # Early return, a strength of 0 causes problems
    # when finding the interpolation strength;
    if strength == 0:
        return COLOR_VISION_DEFICIENCY_MATRICES[deficiency][0.0]
    # If strength is 0.76, we obtain the matrices for 0.7 and 0.8, and interpolate them with strength 0.6 and 0.4;
    matrix_1, matrix_2 = _find_closest_color_deficiency_matrices(deficiency=deficiency, strength=strength)
    # If the strength is 0.761, the interpolation strength is 0.61;
    interpolation_strength: float = np.modf(strength * 10)[0]
    return matrix_1 * interpolation_strength + matrix_2 * (1 - interpolation_strength)


def convert_color_to_deficiency(
    color: str, deficiency: Literal["protanomaly", "deuteranomaly", "tritanomaly"], strength: float = 1
) -> str:
    """
    Given a color expressed as a hex string, convert it into the corresponding color
    as seen by a person with the specified color vision deficiency.

    :param color: the input color, as hex string.
    :param deficiency: the name of the color vision deficiency,
        must be one of "protanomaly", "deuteranomaly", "tritanomaly".
    :param strength: the strength of the color vision deficiency, must be in [0, 1].
        If the strength is a decimal such as [0.0, 0.1, 0.2, ... 0.9, 1.0], use the color deficiency matrix
        provided in "A Physiologically-based Model for Simulation of Color Vision Deficiency", Machado et al. 2009.
        Otherwise, do a linear interpolation between the two closest matrices, as suggested in the paper.
    :return: the output color, as hex string.
    """
    color_deficiency_matrix: Float[np.ndarray, "3 3"]
    if deficiency not in COLOR_VISION_DEFICIENCY_MATRICES:
        raise ValueError(f"❌ unknown color deficiency {deficiency}")
    if strength < 0 or strength > 1:
        raise ValueError(f"❌ strength must be in [0, 1], not {strength}")
    if strength not in COLOR_VISION_DEFICIENCY_MATRICES[deficiency]:
        color_deficiency_matrix = _interpolate_color_deficiency_matrices(deficiency=deficiency, strength=strength)
    else:
        color_deficiency_matrix = COLOR_VISION_DEFICIENCY_MATRICES[deficiency][strength]
    rgb_color = np.array(to_rgb(color))
    assert rgb_color.shape == (3,), f"❌ color {color} has shape {rgb_color.shape}, not (3,)"
    corrected_color = np.clip(rgb_color @ color_deficiency_matrix.T, 0, 1)
    return str(to_hex(corrected_color))


def convert_colors_to_deficiency(
    colors: list[str], deficiency: Literal["protanomaly", "deuteranomaly", "tritanomaly"], strength: float = 1
) -> list[str]:
    """
    Given a list of colors expressed as hex strings, convert them into the corresponding colors
    as seen by a person with the specified color vision deficiency.

    :param colosr: the input colors, as a list of hex strings.
    :param deficiency: the name of the color vision deficiency,
        must be one of "protanomaly", "deuteranomaly", "tritanomaly".
    :param strength: the strength of the color vision deficiency, must be in [0, 1].
        If the strength is a decimal such as [0.0, 0.1, 0.2, ... 0.9, 1.0], use the color deficiency matrix
        provided in "A Physiologically-based Model for Simulation of Color Vision Deficiency", Machado et al. 2009.
        Otherwise, do a linear interpolation between the two closest matrices, as suggested in the paper.
    :return: the output color, as a list of hex strings.
    """
    color_deficiency_matrix: Float[np.ndarray, "3 3"]
    if deficiency not in COLOR_VISION_DEFICIENCY_MATRICES:
        raise ValueError(f"❌ unknown color deficiency {deficiency}")
    if strength < 0 or strength > 1:
        raise ValueError(f"❌ strength must be in [0, 1], not {strength}")
    if strength not in COLOR_VISION_DEFICIENCY_MATRICES[deficiency]:
        color_deficiency_matrix = _interpolate_color_deficiency_matrices(deficiency=deficiency, strength=strength)
    else:
        color_deficiency_matrix = COLOR_VISION_DEFICIENCY_MATRICES[deficiency][strength]
    rgb_colors = np.array([to_rgb(c) for c in colors])
    assert rgb_colors.shape == (
        len(colors),
        3,
    ), f"❌ colors {colors} have shape {rgb_colors.shape}, not {(len(colors), 3)}"
    corrected_colors = np.clip(rgb_colors @ color_deficiency_matrix.T, 0, 1)
    return [str(to_hex(c)) for c in corrected_colors]
