import numpy as np
import pytest
from jaxtyping import Float, Integer
from sklearn.metrics import f1_score, precision_score

from segretini_matplottini.plot import (
    f1,
    false_negatives,
    false_positives,
    precision,
    precision_recall,
    recall,
    roc,
    true_negatives,
    true_positives,
)

from .utils import reset_plot_style, save_tmp_plot  # noqa: F401


@pytest.fixture
def data() -> tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]:
    """
    Generate a fake binary classification result, consisting in a binary array of targets
    and a continuous array of logits. The targets are corrupted with an gamma distribution,
    to create the typical U-curve shape of the logits of a trained binary classifier.
    """
    number_of_samples: int = 1000
    seed: int = 419
    gamma_shape: float = 0.5
    gamma_scale: float = 0.5
    rng = np.random.default_rng(seed)
    # Generate an array of zeros and ones, randomly sampled. These are the targets;
    targets = rng.integers(low=0, high=1, size=number_of_samples, endpoint=True)
    # Generate an array of logits, with score between zero and one. These are the predictions.
    # Corrupt the targets with samples from an gamma distribution,
    # to create the typical U-curve shape of the logits of a trained binary classifier;
    logits = targets.astype(float).copy()
    logits[targets == 0] += rng.gamma(shape=gamma_shape, scale=gamma_scale, size=np.sum(targets == 0))
    logits[targets == 1] -= rng.gamma(shape=gamma_shape, scale=gamma_scale, size=np.sum(targets == 1))
    logits = np.clip(logits, 0, 1)
    return logits, targets


def test_true_positives(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = true_positives(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == (targets == 1).sum()
    assert ax.lines[0].get_ydata()[-1] == 0


def test_true_negatives(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = true_negatives(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == 0
    assert ax.lines[0].get_ydata()[-1] == (targets == 0).sum()


def test_false_positives(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = false_positives(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == (targets == 0).sum()
    assert ax.lines[0].get_ydata()[-1] == 0


def test_false_negatives(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = false_negatives(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == 0
    assert ax.lines[0].get_ydata()[-1] == (targets == 1).sum()


def test_precision(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = precision(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == precision_score(targets, logits >= 0)
    assert ax.lines[0].get_ydata()[-1] == precision_score(y_true=targets, y_pred=logits >= 1, zero_division=0)


def test_recall(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = recall(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == 1
    assert ax.lines[0].get_ydata()[-1] == 0


def test_auc(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = roc(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == 0
    assert ax.lines[0].get_ydata()[-1] == 1


def test_f1(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = f1(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == f1_score(targets, logits >= 0)
    assert ax.lines[0].get_ydata()[-1] == f1_score(y_true=targets, y_pred=logits >= 1, zero_division=0)


def test_precision_recall(data: tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]) -> None:
    logits, targets = data
    _, ax = precision_recall(logits, targets)
    assert ax.lines[0].get_xdata()[0] == 0
    assert ax.lines[0].get_xdata()[-1] == 1
    assert ax.lines[0].get_ydata()[0] == 0
    assert ax.lines[0].get_ydata()[-1] == precision_score(y_true=targets, y_pred=logits >= 0)
