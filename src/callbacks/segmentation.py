"""Segmentation examples callbacks."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from src.visualization import plot_segmentation_results
from src.metrics.results import SegmentationResult
from src.callbacks.base import BaseExamplesPlotterCallback

from typing import Callable


class SegmentationExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def __init__(
        self,
        name: str,
        stages: list[str] | str,
        labels: list[str],
        inverse_preprocessing: Callable,
        cmap: list[tuple[int, int, int]],
    ):
        super().__init__(name, stages)
        self.labels = labels
        self.inverse_preprocessing = inverse_preprocessing
        self.cmap = cmap

    def plot_example_results(
        self, trainer: Trainer, results: SegmentationResult, filepath: str
    ):
        inverse_processing = (
            trainer.datamodule.transform.inference.inverse_preprocessing
        )
        plot_segmentation_results(results, self.cmap, inverse_processing, filepath)
