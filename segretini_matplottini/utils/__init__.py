from segretini_matplottini.utils.colors import (  # noqa: F401
    extend_palette,
    hex_color_to_grayscale,
)
from segretini_matplottini.utils.data import (  # noqa: F401
    compute_speedup,
    compute_speedup_df,
    correct_speedup_df,
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
    add_labels,
    assemble_output_directory_name,
    fix_label_length,
    get_exp_label,
    reset_plot_style,
    save_plot,
    update_bars_width,
)
