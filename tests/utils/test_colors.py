import numpy as np

from segretini_matplottini.utils.colors import (
    COLOR_VISION_DEFICIENCY_MATRICES,
    _find_closest_color_deficiency_matrices,
    _interpolate_color_deficiency_matrices,
    convert_color_to_deficiency,
    convert_color_to_grayscale,
    convert_colors_to_deficiency,
    convert_colors_to_grayscale,
    create_hex_palette,
    extend_palette,
)


def test_create_hex_palette() -> None:
    palette = create_hex_palette("#000000", "#FFFFFF", 2)
    assert palette == ["#000000", "#ffffff"]
    palette = create_hex_palette("#000000", "#FFFFFF", 3)
    assert palette == ["#000000", "#808080", "#ffffff"]


def test_extend_palette() -> None:
    palette = ["#000000", "#FFFFFF"]
    extended_palette = extend_palette(palette, 2)
    assert extended_palette == ["#000000", "#FFFFFF"]
    extended_palette = extend_palette(palette, 3)
    assert extended_palette == ["#000000", "#FFFFFF", "#000000"]
    extended_palette = extend_palette(palette, 1)
    assert extended_palette == ["#000000"]
    extended_palette = extend_palette(palette, 0)
    assert extended_palette == []
    extended_palette = extend_palette(palette, 7)
    assert extended_palette == palette * 3 + ["#000000"]


def test_convert_color_to_grayscale() -> None:
    color = "#FF0000"
    grayscale_color = convert_color_to_grayscale(color, "hsv")
    assert grayscale_color == "#ffffff"
    grayscale_color = convert_color_to_grayscale(color, "hls")
    assert grayscale_color == "#808080"


def test_convert_colors_to_grayscale() -> None:
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    grayscale_colors = convert_colors_to_grayscale(colors, "hsv")
    assert grayscale_colors == ["#ffffff", "#ffffff", "#ffffff"]
    grayscale_colors = convert_colors_to_grayscale(colors, "hls")
    assert grayscale_colors == ["#808080", "#808080", "#808080"]


def test_find_closest_color_deficiency_matrices() -> None:
    m1, m2 = _find_closest_color_deficiency_matrices("protanomaly", 0.03)
    assert np.allclose(m1, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.0])
    assert np.allclose(m2, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.1])
    m1, m2 = _find_closest_color_deficiency_matrices("protanomaly", 0.13)
    assert np.allclose(m1, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.1])
    assert np.allclose(m2, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.2])
    m1, m2 = _find_closest_color_deficiency_matrices("protanomaly", 0.97)
    assert np.allclose(m1, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.9])
    assert np.allclose(m2, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][1.0])


def test_interpolate_color_deficiency_matrices() -> None:
    m = _interpolate_color_deficiency_matrices("protanomaly", 0.05)
    assert np.allclose(
        m,
        (COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.0] + COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.1])
        / 2,
    )
    m = _interpolate_color_deficiency_matrices("protanomaly", 0.64)
    assert np.allclose(
        m,
        (
            COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.6] * 0.4
            + COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.7] * 0.6
        ),
    )
    m = _interpolate_color_deficiency_matrices("protanomaly", 0.0)
    assert np.allclose(m, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.0])
    m = _interpolate_color_deficiency_matrices("protanomaly", 0.9)
    assert np.allclose(m, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][0.9])
    m = _interpolate_color_deficiency_matrices("protanomaly", 1)
    assert np.allclose(m, COLOR_VISION_DEFICIENCY_MATRICES["protanomaly"][1])


def test_convert_color_to_deficiency() -> None:
    color = "#E8CFB5"
    deficient_color = convert_color_to_deficiency(color, "protanomaly", 1)
    assert deficient_color == "#d8cfb4"


def test_convert_colorw_to_deficiency() -> None:
    colors = ["#E8CFB5", "#81C798", "#358787", "#358787"]
    deficient_colors = convert_colors_to_deficiency(colors, "protanomaly", 1)
    assert deficient_colors == ["#d8cfb4", "#c6ba96", "#7b7e87", "#7b7e87"]

    colors = ["#FF0000"]
    deficient_colors = convert_colors_to_deficiency(colors, "tritanomaly", 1)
    assert deficient_colors == ["#ff0001"]
