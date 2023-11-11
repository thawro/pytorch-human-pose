"""Metrics plotting functions."""

import matplotlib.pyplot as plt

from src.metrics.storage import MetricsStorage


def plot_metrics(metrics_storage: MetricsStorage, filepath: str | None) -> None:
    """Plot metrics for each split and step"""
    metrics = metrics_storage.metrics
    ncols = len(metrics)

    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
    for (metric_name, split_values), ax in zip(metrics.items(), axes):
        for split, values in split_values.items():
            epochs = list(range(len(values)))
            ax.plot(epochs, values, label=split)
            ax.scatter(epochs, values)
            ax.legend()
            ax.set_title(metric_name, fontsize=18)
            ax.set_xlabel("Epoch", fontsize=14)
    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()
