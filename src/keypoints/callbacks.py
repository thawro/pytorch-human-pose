"""Dummy results callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING
from .results import SPPEKeypointsResults, MPPEKeypointsResult
from .visualization import plot_sppe_results_heatmaps, plot_mppe_results_heatmaps
from src.base.callbacks import BaseExamplesPlotterCallback


if TYPE_CHECKING:
    from src.base.trainer import Trainer


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def __init__(self, name: str | None, det_thr: float = 0.2):
        super().__init__(name)
        self.det_thr = det_thr

    def plot_example_results(
        self,
        trainer: Trainer,
        results: SPPEKeypointsResults | list[MPPEKeypointsResult],
        filepath: str,
    ):
        limbs = trainer.datamodule.train_ds.limbs
        if isinstance(results, SPPEKeypointsResults):
            plot_sppe_results_heatmaps(results, limbs, filepath, thr=self.det_thr)
        else:
            plot_mppe_results_heatmaps(results, filepath)
