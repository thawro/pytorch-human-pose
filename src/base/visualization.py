"""Metrics plotting functions."""

from typing import Literal

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.base.storage import MetricsStorage


def plot_metrics_matplotlib(
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


def plot_metrics_plotly(
    metrics_storage: MetricsStorage,
    step_name: Literal["step", "epoch"],
    filepath: str | None,
    tickformat: str = ".5f",
) -> None:
    palette = px.colors.qualitative.T10
    """Plot metrics for each split and step"""
    _metrics = metrics_storage.metrics
    metrics = _metrics.copy()
    metrics.pop("epoch")
    metrics.pop("step")
    ncols = len(metrics)
    titles = list(metrics.keys())
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles)

    for i, (metric_name, split_values) in enumerate(metrics.items()):
        for j, (split, logged_values) in enumerate(split_values.items()):
            if "sanity" in split:  # dont plot sanity check metrics
                continue
            c = palette[j]
            steps = [logged[step_name] for logged in logged_values]
            values = [logged["value"] for logged in logged_values]
            fig.add_trace(
                go.Scatter(x=steps, y=values, name=split, marker=dict(color=c)), row=1, col=i + 1
            )
        fig.update_xaxes(title_text=metrics_storage.name, row=1, col=i)
    base_h = 600
    fig.update_layout(
        height=base_h, width=base_h * ncols + base_h // 4, yaxis=dict(tickformat=tickformat)
    )
    if filepath is not None:
        fig.write_html(filepath)
