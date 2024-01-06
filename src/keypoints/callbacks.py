"""Dummy results callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING
from .results import SPPEKeypointsResults, MPPEKeypointsResults
from .visualization import plot_sppe_results_heatmaps, plot_mppe_results_heatmaps
from src.base.callbacks import BaseExamplesPlotterCallback


if TYPE_CHECKING:
    from src.base.trainer import Trainer


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def plot_example_results(
        self,
        trainer: Trainer,
        results: SPPEKeypointsResults | MPPEKeypointsResults,
        filepath: str,
    ):
        limbs = trainer.datamodule.train_ds.limbs
        if isinstance(results, SPPEKeypointsResults):
            plot_sppe_results_heatmaps(results, limbs, filepath)
        else:
            plot_mppe_results_heatmaps(results, limbs, filepath)
