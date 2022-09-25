from functools import reduce
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as st
from scipy.stats.mstats import gmean


def remove_outliers_ci(data: Union[pd.Series, list[float], np.ndarray], sigmas: float = 3):
    """
    Filter a sequence of data by keeping only values within "sigma" standard deviations from the mean.
    This is a simple way to filter outliers, it is more useful to clean data
    for visualizations than for sound statistical analyses.

    :param data: A 1D sequence of numerical data, iterable.
    :param sigmas: Number of standard deviations outside which a value is consider to be an outlier.
    :return: Data without outliera.
    """
    return data[np.abs(st.zscore(data)) < sigmas]


def remove_outliers_iqr(data, quantile: float = 0.75):
    """
    Filter a sequence of data by removing outliers looking at the quantiles of the distribution.
    Find quantiles (by default, `Q1` and `Q3`), and interquantile range (by default, `Q3 - Q1`),
    and keep values in `[Q1 - iqr_extension * IQR, Q3 + iqr_extension * IQR]`.
    This is the same range used to identify whiskers in a boxplot (e.g. in Pandas and Seaborn).

    :param data: A sequence of numerical data, iterable.
    :param quantile: Upper quantile value used as filtering threshold.
        Also use `(1 - quantile)` as lower threshold. Should be in `[0.5, 1]`.
    :return: Data without outliers.
    """
    assert quantile >= 0.5 and quantile <= 1
    q1 = np.quantile(data, 1 - quantile)
    q3 = np.quantile(data, quantile)
    iqr = scipy.stats.iqr(data, rng=(100 - 100 * quantile, 100 * quantile))
    return data[(data >= q1 - iqr * q1) & (data <= q3 + iqr * q3)]


def _remove_outliers_from_dataframe(
    data: pd.DataFrame,
    column: str,
    remove_outliers_func: Callable,
    groupby: Optional[list[str]] = None,
    reset_index: bool = True,
    drop_index: bool = True,
    debug: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Filter a pandas DataFrame by removing outliers from a specified column.
    All rows with outlier values are removed.
    Outliers are identified using the specified `remove_outliers_func`.

    :param data: a pandas DataFrame.
    :param column: Name of the column on which the outlier detection is performed.
    :param remove_outliers_func: Function used to remove outliers on an individual column.
    :param groupby: If not None, perform outlier detection on the data after grouping it
        by the specified set of columns.
    :param reset_index: If True, reset the index after filtering.
    :param drop_index: If True, drop the original index column after reset.
    :param sigmas: Number of standard deviations outside which a value is consider to be an outlier.
    :param debug: If True, print how many outliers have been removed.
    :param kwargs: Additional keywork arguments provided to `remove_outliers_func`.
    :return: Data without outliers.
    """

    def _remove_outliers_df(data: pd.DataFrame, column, reset_index, drop_index, remove_outliers_func, **kwargs):
        col = data[column]
        res = data.loc[remove_outliers_func(col, **kwargs).index]
        if reset_index:
            res = res.reset_index(drop=drop_index)
        return res

    old_len = len(data)
    if groupby is None:
        new_data = _remove_outliers_df(data, column, reset_index, drop_index, remove_outliers_func, **kwargs)
    else:
        filtered = []
        for _, g in data.groupby(groupby, sort=False):
            filtered += [_remove_outliers_df(g, column, reset_index, drop_index, remove_outliers_func, **kwargs)]
        new_data = pd.concat(filtered, ignore_index=True)
    if debug and (len(new_data) < old_len):
        print(f"removed {old_len - len(new_data)} outliers")
    return new_data


def remove_outliers_from_dataframe_ci(
    data: pd.DataFrame,
    column: str,
    groupby: Optional[list[str]] = None,
    sigmas: float = 3,
    reset_index: bool = True,
    drop_index: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Filter a pandas DataFrame by removing outliers from a specified column.
    All rows with outlier values are removed.
    Outliers are identified as being outside "sigma" standard deviations from the column mean.
    This is a simple way to filter outliers, it is more useful to clean data
    for visualizations than for sound statistical analyses.

    :param data: a pandas DataFrame.
    :param column: Name of the column on which the outlier detection is performed.
    :param remove_outliers_func: Function used to remove outliers on an individual column.
    :param groupby: If not None, perform outlier detection on the data after grouping it
        by the specified set of columns.
    :param sigmas: Number of standard deviations outside which a value is consider to be an outlier.
    :param reset_index: If True, reset the index after filtering.
    :param drop_index: If True, drop the original index column after reset.
    :param debug: If True, print how many outliers have been removed.
    :param kwargs: Additional keywork arguments provided to `remove_outliers_func`.
    :return: Data without outliers.
    """
    return _remove_outliers_from_dataframe(
        data,
        column,
        remove_outliers_func=remove_outliers_ci,
        groupby=groupby,
        reset_index=reset_index,
        drop_index=drop_index,
        debug=debug,
        sigmas=sigmas,
    )


def remove_outliers_from_dataframe_iqr(
    data: pd.DataFrame,
    column: str,
    groupby: Optional[list[str]] = None,
    quantile: float = 0.75,
    reset_index: bool = True,
    drop_index: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Filter a pandas DataFrame by removing outliers from a specified column.
    All rows with outlier values are removed.
    Outliers are identified as being outside "sigma" standard deviations from the column mean.
    This is a simple way to filter outliers, it is more useful to clean data
    for visualizations than for sound statistical analyses.

    :param data: a pandas DataFrame.
    :param column: Name of the column on which the outlier detection is performed.
    :param remove_outliers_func: Function used to remove outliers on an individual column.
    :param groupby: If not None, perform outlier detection on the data after grouping it
        by the specified set of columns.
    :param quantile: Upper quantile value used as filtering threshold.
        Also use `(1 - quantile)` as lower threshold. Should be in `[0.5, 1]`.
    :param reset_index: If True, reset the index after filtering.
    :param drop_index: If True, drop the original index column after reset.
    :param debug: If True, print how many outliers have been removed.
    :param kwargs: Additional keywork arguments provided to `remove_outliers_func`.
    :return: Data without outliers.
    """
    return _remove_outliers_from_dataframe(
        data,
        column,
        remove_outliers_func=remove_outliers_iqr,
        groupby=groupby,
        reset_index=reset_index,
        drop_index=drop_index,
        debug=debug,
        quantile=quantile,
    )


def compute_speedup(data: pd.DataFrame, col_slow: str, col_fast: str, col_speedup: str) -> pd.DataFrame:
    """
    Add a column to a dataframe that represents a speedup,
    and "col_slow", "col_fast" are execution times (e.g. CPU and GPU execution time).
    Speedup is computed as `data[col_slow] / data[col_fast]`

    :param data: A Pandas DataFrame where the speedup is computed.
    :param col_slow: The baseline column used for the speedup.
    :param col_fast: The other colum used for the speedup.
    :param col_speedup: Name of the column where the speedup is stored.
    :return: The DataFrame.
    """
    data[col_speedup] = data[col_slow] / data[col_fast]
    return data


def correct_speedup_df(
    data: pd.DataFrame,
    groupby: list[str],
    baseline_filter_col: str,
    baseline_filter_val: str,
    speedup_col_name: str = "speedup",
    speedup_col_name_reference: Optional[str] = None,
):
    """
    Divide the speedups in `speedup_col_name` by the geomean of `speedup_col_name_reference`,
    grouping values by the columns in `groupby` and specifying a baseline column and value to use as reference.
    In most cases, `speedup_col_name` and `speedup_col_name_reference` are the same value.
    Useful to ensure that the geomean baseline speedup is 1, and that the other speedups are corrected to reflect that.

    1. Divide the data in groups denoted by `groupby`
    2. For each group, select rows where `data[baseline_filter_col] == baseline_filter_val`
    3. Compute the geometric mean of the column `speedup_col_name_reference` for the rows selected at (2)
    4. Divide the values in `speedup_col_name_reference` for the current group selected at (1)
       by the geometric mean computed at (3)

    :param data: Input DataFrame.
    :param groupby: List of columns on which the grouping is performed, e.g. `["benchmark_name", "implementation"]`.
    :param baseline_filter_col: One or more columns used to recognize the baseline, e.g. `["hardware"]`.
    :param baseline_filter_val : One or more values in `baseline_filter_col` used
        to recognize the baseline, e.g. `["cpu"]`.
    :param speedup_col_name: Name of the speedup column to adjust. The default is `"speedup"`.
    :param speedup_col_name_reference: Name of the reference speedup column,
        by default it is the same as `"speedup_col_name"`.
    :return: The updated DataFrame.
    """
    if not speedup_col_name_reference:
        speedup_col_name_reference = speedup_col_name
    for _, g in data.groupby(groupby):
        gmean_speedup = gmean(g.loc[g[baseline_filter_col] == baseline_filter_val, speedup_col_name_reference])
        data.loc[g.index, speedup_col_name] /= gmean_speedup
    return data


def compute_speedup_df(
    data: pd.DataFrame,
    groupby: list[str],
    baseline_filter_col: list[str],
    baseline_filter_val: list[str],
    speedup_col_name: str = "speedup",
    time_column_name: str = "exec_time",
    baseline_col_name: str = "baseline_time",
    correction: bool = True,
    aggregation: Callable = np.median,
    compute_relative_perf: bool = False,
):
    """
    Compute speedups on a DataFrame by grouping values.

    1. Divide the data in groups denoted by `groupby`
    2. For each group, select rows where `data[baseline_filter_col] == baseline_filter_val`
    3. Compute the mean of the column `speedup_col_name_reference` for the rows selected at (2)
    4. Divide the values in `speedup_col_name_reference` for the current group selected at (1)
       by the mean computed at (3)

    :param data: Input DataFrame.
    :param groupby: List of columns on which the grouping is performed, e.g. `["benchmark_name", "implementation"]`.
    :param baseline_filter_col: One or more columns used to recognize the baseline, e.g. `["hardware"]`.
    :param baseline_filter_val : One or more values in `baseline_filter_col` used
        to recognize the baseline, e.g. `["cpu"]`.
    :param speedup_col_name: Name of the speedup column to adjust. The default is `"speedup"`.
    :param time_column_name: Name of the execution time column. The default is `"exec_time"`. This is the column
        where the mean performance is computed, and speedup is obtained as a relative execution time.
    :param baseline_col_name: Add a new column where we add the execution time of the baseline used to compute the
        speedup in each group. The default is `"baseline_time"`.
    :param correction : If `True`, ensure that the median of the baseline is 1. The default is `True`.
    :param aggregation: Function used to aggregate values. The default is `np.median`.
    :param compute_relative_perf: If `True`, compute relative performance instead of speedup (i.e. `1 / speedup`);
    :return: The updated DataFrame.
    """

    # Initialize speedup values;
    data[speedup_col_name] = 1
    data[baseline_col_name] = 0

    if type(baseline_filter_col) is not list:
        baseline_filter_col = [baseline_filter_col]
    if type(baseline_filter_val) is not list:
        baseline_filter_val = [baseline_filter_val]

    assert len(baseline_filter_col) == len(baseline_filter_val)

    grouped_data = data.groupby(groupby, as_index=False)
    for _, group in grouped_data:
        # Compute the median baseline computation time;
        indices = [group[group[i] == j].index for i, j in zip(baseline_filter_col, baseline_filter_val)]
        reduced_index = reduce(lambda x, y: x.intersection(y), indices)
        mean_baseline = aggregation(data.loc[reduced_index, time_column_name])
        # Compute the speedup for this group;
        group.loc[:, speedup_col_name] = (
            (group[time_column_name] / mean_baseline)
            if compute_relative_perf
            else (mean_baseline / group[time_column_name])
        )
        group.loc[:, baseline_col_name] = mean_baseline
        data.loc[group.index, :] = group

        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        if correction:
            gmean_speedup = gmean(data.loc[reduced_index, speedup_col_name])
            group.loc[:, speedup_col_name] /= gmean_speedup
            data.loc[group.index, :] = group
