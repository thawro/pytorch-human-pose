"""Dummy results callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING
from src.metrics.results import KeypointsResult

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer


from src.visualization import plot_heatmaps
from src.callbacks.base import BaseExamplesPlotterCallback


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def plot_example_results(
        self, trainer: Trainer, results: KeypointsResult, filepath: str
    ):
        plot_heatmaps(results, filepath)
