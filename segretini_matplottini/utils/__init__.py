from segretini_matplottini.utils.colors import (  # noqa: F401
    create_hex_palette,
    extend_palette,
    hex_color_to_grayscale,
)
from segretini_matplottini.utils.data import (  # noqa: F401
    compute_relative_performance,
    find_outliers_right_quantile,
    get_ci_size,
    get_upper_ci_size,
    remove_outliers_ci,
    remove_outliers_from_dataframe_ci,
    remove_outliers_from_dataframe_iqr,
    remove_outliers_iqr,
)
from segretini_matplottini.utils.legend import (  # noqa: F401
    add_legend_with_dark_shadow,
    transpose_legend_labels,
)
from segretini_matplottini.utils.plot import (  # noqa: F401
    activate_dark_background,
    add_arrow_to_barplot,
    add_labels,
    add_labels_to_bars,
    assemble_filenames_to_save_plot,
    fix_label_length,
    get_exp_label,
    get_labels_for_bars,
    reset_plot_style,
    save_plot,
    update_bars_width,
)
