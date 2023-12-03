"""Dummy results callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING
from .results import SPPEKeypointsResults
from .visualization import plot_results_heatmaps
from src.base.callbacks import BaseExamplesPlotterCallback


if TYPE_CHECKING:
    from src.base.trainer import Trainer


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def plot_example_results(
        self, trainer: Trainer, results: SPPEKeypointsResults, filepath: str
    ):
        limbs = trainer.datamodule.train_ds.limbs
        plot_results_heatmaps(results, limbs, filepath)
