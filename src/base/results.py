from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.utils.image import make_grid


@dataclass
class BaseResult:
    def set_preds(self):
        pass

    def plot(self) -> dict[str, np.ndarray]:
        raise NotImplementedError()


def plot_results(
    results: list[BaseResult], plot_name: str, filepath: str | None = None
) -> np.ndarray:
    n_rows = min(20, len(results))
    grids = []
    for i in range(n_rows):
        result = results[i]
        result.set_preds()
        plots = result.plot()
        result_plot = plots[plot_name]
        grids.append(result_plot)
    final_grid = make_grid(grids, nrows=len(grids), pad=20)
    if filepath is not None:
        im = Image.fromarray(final_grid)
        im.save(filepath)
    return final_grid
