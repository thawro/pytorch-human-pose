"""Dummy results callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from src.base.callbacks import BaseCallback, BaseExamplesPlotterCallback, DatasetExamplesCallback
from src.utils.image import make_grid

from .results import KeypointsResult

if TYPE_CHECKING:
    from .trainer import KeypointsTrainer


def plot_results(results: list["KeypointsResult"], filepath: str | None = None) -> np.ndarray:
    n_rows = min(20, len(results))
    grids = []
    for i in range(n_rows):
        result = results[i]
        result.set_preds()
        result_plot = result.plot()
        grids.append(result_plot)
    final_grid = make_grid(grids, nrows=len(grids), pad=20)
    if filepath is not None:
        im = Image.fromarray(final_grid)
        im.save(filepath)
    return final_grid


class KeypointsExamplesPlotterCallback(BaseExamplesPlotterCallback):
    """Plot prediction examples"""

    def __init__(self, name: str | None, det_thr: float = 0.1):
        super().__init__(name)
        self.det_thr = det_thr

    def plot_example_results(
        self, trainer: KeypointsTrainer, results: list[KeypointsResult], filepath: str
    ):
        plot_results(results, filepath)


class KeypointsDatasetExamplesCallback(DatasetExamplesCallback):
    def __init__(
        self,
        splits: list[Literal["train", "val", "test"]] = ["train"],
        n: int = 10,
        random_idxs: bool = False,
    ) -> None:
        super().__init__(splits, n, random_idxs)
