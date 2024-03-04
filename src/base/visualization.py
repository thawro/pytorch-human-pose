"""Metrics plotting functions."""

from typing import Literal

import matplotlib.pyplot as plt

from src.base.storage import MetricsStorage


def plot_metrics(
    metrics_storage: MetricsStorage,
    step_name: Literal["step", "epoch"],
    filepath: str | None,
) -> None:
    """Plot metrics for each split and step"""
    _metrics = metrics_storage.metrics
    metrics = _metrics.copy()
    metrics.pop("epoch")
    metrics.pop("step")
    ncols = len(metrics)

    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
    if ncols == 1:
        axes = [axes]
    for (metric_name, split_values), ax in zip(metrics.items(), axes):
        for split, logged_values in split_values.items():
            if "sanity" in split:  # dont plot sanity check metrics
                continue
            steps = [logged[step_name] for logged in logged_values]
            values = [logged["value"] for logged in logged_values]
            ax.plot(steps, values, label=split)
            ax.scatter(steps, values)
            ax.legend()
            ax.set_title(metric_name, fontsize=18)
            ax.set_xlabel(metrics_storage.name, fontsize=14)
    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()
