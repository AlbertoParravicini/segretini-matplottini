from pathlib import Path

import numpy as np
from jaxtyping import Float, Integer
from sklearn.metrics import (
    f1_score,
)

from segretini_matplottini.plot.binary_classification import binary_classification
from segretini_matplottini.utils import (
    assemble_filenames_to_save_plot,
    save_plot,
)

#########
# Setup #
#########

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

#############
# Load data #
#############


def generate_data(
    number_of_samples: int = 1000, seed: int = 419, gamma_shape: float = 0.5, gamma_scale: float = 0.5
) -> tuple[Float[np.ndarray, "#n"], Integer[np.ndarray, "#n"]]:
    """
    Generate a fake binary classification result, consisting in a binary array of targets
    and a continuous array of logits. The targets are corrupted with an gamma distribution,
    to create the typical U-curve shape of the logits of a trained binary classifier.
    The default settings return a result with F1 score of 0.8363, when measured with threshold 0.5.

    :param number_of_samples: Number of samples to generate.
    :param seed: Seed for the random number generator.
    :param gamma_shape: Shape parameter of the gamma distribution.
    :param gamma_scale: Scale parameter of the gamma distribution.
    :return: A tuple of two arrays, the first is the logits, the second is the targets.
    """
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
    f1 = f1_score(targets, logits > 0.5)
    print(f"👉 generated a random binary classification problem with {f1:.4f} F1 score")
    return logits, targets


if __name__ == "__main__":
    logits, targets = generate_data()
    binary_classification(logits, targets)
    save_plot(
        assemble_filenames_to_save_plot(
            directory=PLOT_DIR,
            plot_name="binary_classification",
            add_timestamp_prefix_to_plot_name=False,
            store_plot_into_timestamp_subfolder=False,
        ),
        verbose=True,
    )
