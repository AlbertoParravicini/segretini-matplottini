import functools
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pytest

from segretini_matplottini.utils import save_plot


@pytest.fixture(autouse=True)
def reset_plot_style() -> None:
    # Reset plotting settings
    plt.rcdefaults()
    return


def save_tmp_plot(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)  # Necessary to combine pytest's fixtures and the custom decorator
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        # Save the plot to a temporary file;
        plot_dir = Path(__file__).parent.parent.parent / "plots" / "tests"
        plot_dir.mkdir(parents=True, exist_ok=True)
        caller_name = func.__name__
        save_plot(plot_dir / f"{Path(__file__).stem}_{caller_name}.png")
        return result

    return wrapper
