"""Dummy results callbacks."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

from PIL import Image

from src.base.callbacks import BaseCallback, BaseExamplesPlotterCallback, DatasetExamplesCallback

from .results import KeypointsResult
from .visualization import plot_mppe_results_heatmaps

if TYPE_CHECKING:
    from .trainer import KeypointsTrainer


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def __init__(self, name: str | None, det_thr: float = 0.1):
        super().__init__(name)
        self.det_thr = det_thr

    def plot_example_results(
        self, trainer: KeypointsTrainer, results: list[KeypointsResult], filepath: str
    ):
        plot_mppe_results_heatmaps(results, filepath)


class KeypointsDatasetExamplesCallback(DatasetExamplesCallback):
    def __init__(
        self,
        splits: list[Literal["train", "val", "test"]] = ["train"],
        n: int = 10,
        random_idxs: bool = False,
    ) -> None:
        super().__init__(splits, n, random_idxs)
