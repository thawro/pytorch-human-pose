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
    horizontal: bool = True,
) -> None:
    """Plot metrics for each split and step"""
    palette = px.colors.qualitative.T10
    _metrics = metrics_storage.metrics
    metrics = _metrics.copy()
    metrics.pop("epoch")
    metrics.pop("step")
    num_metrics = len(metrics)
    titles = list(metrics.keys())
    h, w = 500, 600

    if horizontal:
        rows, cols = 1, num_metrics
    else:
        rows, cols = num_metrics, 1

    single_split_keys = [k for k, v in metrics.items() if len(v) == 1]
    multi_split_keys = [k for k, v in metrics.items() if len(v) > 1]
    showlegend = None
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for i, (metric_name, split_values) in enumerate(metrics.items()):
        if showlegend is not None and showlegend:
            showlegend = False
        if showlegend is None and metric_name in multi_split_keys:
            # _showlegend = True
            showlegend = True
        tickformat = ".6f"
        if horizontal:
            row, col = 1, i + 1
        else:
            row, col = i + 1, 1
        for j, (split, logged_values) in enumerate(split_values.items()):
            if "sanity" in split:  # dont plot sanity check metrics
                continue
            c = palette[j]
            steps = [logged[step_name] for logged in logged_values]
            values = [logged["value"] for logged in logged_values]
            # show the legend only for first plot of multi_split metrics

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    name=split,
                    mode="lines+markers",
                    marker=dict(color=c),
                    legendgroup=split,
                    showlegend=showlegend if showlegend else False,
                ),
                row=row,
                col=col,
            )
            if min(values) < 1e-4 or max(values) > 1e4:
                tickformat = ".3e"
        fig.update_yaxes(tickformat=tickformat, row=row, col=col)
        fig.update_xaxes(title_text=step_name, row=row, col=col)

    if horizontal:
        margin = dict(t=100, b=100, l=w // 2, r=w // 2)
        fig_h = h
        fig_w = w * num_metrics
        legend = dict(orientation="v", yanchor="bottom", y=0.5, xanchor="right", x=-0.03)
    else:
        margin = dict(t=100, b=100, l=w // 2, r=w // 2)
        fig_h = h * num_metrics
        fig_w = w
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.5)

    fig_h += margin["t"] + margin["b"]
    fig_w += margin["l"] + margin["r"]

    fig.update_layout(height=fig_h, width=fig_w, margin=margin, legend=legend)

    if filepath is not None:
        fig.write_html(filepath)
