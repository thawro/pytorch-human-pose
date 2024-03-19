"""Metrics plotting functions."""

import math
from typing import Literal

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.base.storage import MetricsStorage, SystemMonitoringStorage


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
    metrics = metrics_storage.metrics.copy()
    metrics.pop("epoch")
    metrics.pop("step")
    num_metrics = len(metrics)
    titles = list(metrics.keys())
    h, w = 500, 600

    if horizontal:
        nrows, ncols = 1, num_metrics
    else:
        nrows, ncols = num_metrics, 1

    single_split_keys = [k for k, v in metrics.items() if len(v) == 1]
    multi_split_keys = [k for k, v in metrics.items() if len(v) > 1]
    showlegend = None
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)
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


def plot_system_monitoring(
    metrics_storage: SystemMonitoringStorage,
    filepath: str | None,
) -> None:
    """Plot metrics for each split and step"""
    palette = px.colors.qualitative.Plotly
    metrics = metrics_storage.metrics.copy()
    timestamps = metrics_storage.timestamps.copy()

    disk_mb_metrics_names = ["disk_mb_available", "disk_mb_used"]
    disk_pct_metrics_names = ["disk_pct_used"]
    cpu_pct_metrics_names = ["cpu_pct_utilisation"]

    # gpu_<idx>_pct_memory_used, gpu_<idx>_mb_memory_used, gpu_<idx>_pct_utilisation

    all_gpus_metrics_names = [metric for metric in metrics.keys() if "gpu_" in metric]
    all_gpus_idxs = list(set([int(name.split("_")[1]) for name in all_gpus_metrics_names]))

    gpu2metrics_names = {
        idx: [name for name in all_gpus_metrics_names if idx == int(name.split("_")[1])]
        for idx in all_gpus_idxs
    }

    first_gpu_idx = all_gpus_idxs[0]
    first_gpu_metrics_names = gpu2metrics_names[first_gpu_idx]

    gpu_metrics_names = {}
    for name in first_gpu_metrics_names:
        metric_name = name.replace(f"_{first_gpu_idx}", "")
        per_gpu_names = [
            name.replace(f"_{first_gpu_idx}", f"_{gpu_idx}") for gpu_idx in all_gpus_idxs
        ]
        gpu_metrics_names[metric_name] = per_gpu_names
    num_plots = 1 + 1 + 1 + len(gpu_metrics_names)

    ncols = 3
    nrows = math.ceil(num_plots / ncols)

    subplots = {
        **gpu_metrics_names,
        "CPU percentage": cpu_pct_metrics_names,
        "Disk megabytes": disk_mb_metrics_names,
        "Dick percentage": disk_pct_metrics_names,
    }

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(subplots.keys()))

    row, col = 1, 1
    color_idx = len(all_gpus_idxs)

    for i, (subplot_title, metric_names) in enumerate(subplots.items()):
        row = (i // ncols) + 1
        col = (i % ncols) + 1
        for i, metric_name in enumerate(metric_names):
            if "gpu" in metric_name:
                gpu_idx = int(metric_name.split("_")[1])
                legendgroup = f"gpu_{gpu_idx}"
                c = palette[gpu_idx]
            else:
                legendgroup = metric_name
                c = palette[color_idx]
                color_idx += 1
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=metrics[metric_name],
                    name=metric_name,
                    marker=dict(color=c),
                    mode="lines+markers",
                    legendgroup=legendgroup,
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Time", row=row, col=col)

    # height2gap_ratio = 1.05
    # base_h = 400
    fig.update_layout(
        legend_tracegroupgap=50,
    )

    if filepath is not None:
        fig.write_html(filepath)
