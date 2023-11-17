"""Dummy plotting functions."""

import matplotlib.pyplot as plt
from .results import DummyResult


def plot_dummy_results(results: DummyResult, filepath: str | None) -> None:
    """Plot preds vs targets"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    ax.plot(results.targets, results.preds)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()
