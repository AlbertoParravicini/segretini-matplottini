import numpy as np
import pandas as pd
import pytest

from segretini_matplottini.utils.data import compute_relative_performance


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat": ["baseline", "model_1", "model_2", "baseline", "model_1", "model_2"],
            "group_by_1": ["A", "A", "A", "B", "B", "B"],
            "val": [0.1, 0.3, 0.6, 2, 4, 6],
        }
    )


def test_relative_performance(data: pd.DataFrame) -> None:
    rel_performance = compute_relative_performance(data, category="cat", baseline_category="baseline", value="val")
    assert np.allclose(
        rel_performance["val_relative_performance"],
        np.array(
            [
                0.09523809523809523,
                0.2857142857142857,
                0.5714285714285714,
                1.9047619047619047,
                3.8095238095238093,
                5.714285714285714,
            ]
        ),
    )
    assert np.allclose(
        rel_performance.groupby("cat").mean(numeric_only=True).val_relative_performance,
        np.array([1.0, 2.0476190476190474, 3.142857142857143]),
    )


def test_relative_performance_with_baseline_col(data: pd.DataFrame) -> None:
    rel_performance = compute_relative_performance(
        data, category="cat", baseline_category="baseline", value="val", add_baseline_value_to_result=True
    )
    assert np.allclose(rel_performance["val_baseline_value"], np.array([1.05]))


def test_relative_performance_with_grouping(data: pd.DataFrame) -> None:
    rel_performance = compute_relative_performance(
        data, category="cat", baseline_category="baseline", value="val", groupby=["group_by_1"]
    )
    assert np.allclose(
        rel_performance["val_relative_performance"],
        np.array([1, 3, 6, 1, 2, 3]),
    )
    assert np.allclose(
        rel_performance.groupby("cat").mean(numeric_only=True).val_relative_performance,
        np.array([1.0, 2.5, 4.5]),
    )


def test_relative_performance_with_grouping_and_baseline_col(data: pd.DataFrame) -> None:
    rel_performance = compute_relative_performance(
        data,
        category="cat",
        baseline_category="baseline",
        value="val",
        groupby=["group_by_1"],
        add_baseline_value_to_result=True,
    )
    assert np.allclose(
        rel_performance["val_baseline_value"],
        np.array([0.1, 0.1, 0.1, 2, 2, 2]),
    )
