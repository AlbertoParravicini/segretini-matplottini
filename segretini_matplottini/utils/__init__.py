from segretini_matplottini.utils.colors import (  # noqa: F401
    convert_color_to_deficiency,
    convert_color_to_grayscale,
    convert_colors_to_deficiency,
    convert_colors_to_grayscale,
    create_hex_palette,
    extend_palette,
)
from segretini_matplottini.utils.data import (  # noqa: F401
    ConfusionMatrix,
    compute_relative_performance,
    confusion_matrix,
    false_negatives,
    false_positives,
    find_outliers_right_quantile,
    get_ci_size,
    get_upper_ci_size,
    remove_outliers_ci,
    remove_outliers_from_dataframe_ci,
    remove_outliers_from_dataframe_iqr,
    remove_outliers_iqr,
    true_negatives,
    true_positives,
)
from segretini_matplottini.utils.legend import (  # noqa: F401
    add_legend_with_dark_shadow,
    get_legend_handles_from_colors,
    transpose_legend_labels,
)
from segretini_matplottini.utils.plot import (  # noqa: F401
    activate_dark_background,
    add_arrow_to_barplot,
    add_labels_to_bars,
    adjust_rows_and_columns_to_number_of_plots,
    assemble_filenames_to_save_plot,
    fix_label_length,
    get_exp_label,
    get_labels_for_bars,
    reset_plot_style,
    save_plot,
    update_bars_width,
)
