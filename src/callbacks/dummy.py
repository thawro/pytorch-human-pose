"""Dummy results callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING
from src.metrics.results import Result

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer


from src.visualization import plot_dummy_results
from src.callbacks.base import BaseExamplesPlotterCallback


class DummyExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def plot_example_results(self, trainer: Trainer, results: Result, filepath: str):
        plot_dummy_results(results, filepath)
