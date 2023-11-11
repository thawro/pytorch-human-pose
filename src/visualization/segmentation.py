"""Segmentation related plotting functions."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from src.metrics.results import SegmentationResult


def plot_segmentation_results(
    results: SegmentationResult,
    cmap: list[tuple[int, int, int]],
    inverse_preprocessing: Callable,
    filepath: str | None,
) -> None:
    """Plot image, y_true and y_pred (masks) for each result."""

    images = inverse_preprocessing(results.images)
    nrows = len(images)
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 7 * nrows))

    for i in range(nrows):
        ax = axes[i]

        pred = results.preds[i].argmax(0)
        target = results.targets[i]
        image = images[i]

        target = colorize_mask(target, cmap=cmap)
        pred = colorize_mask(pred, cmap=cmap)
        ax[0].imshow(image)
        ax[1].imshow(target)
        ax[2].imshow(pred)
        for _ax in ax:
            _ax.set_axis_off()

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_mask_on_image(image: np.ndarray, mask: np.ndarray, txt: str | None = None) -> np.ndarray:
    """Put segmentation mask on image."""
    fontscale = 0.5  # line width
    fw = 1
    alpha = 0.9
    BLACK, WHITE = (0, 0, 0), (255, 255, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cv2.addWeighted(mask, alpha, image, 1.0, 0, image)
    if txt is not None:
        w, h = cv2.getTextSize(txt, 0, fontScale=fontscale, thickness=fw)[0]
        cv2.rectangle(image, (0, 0), (w + h // 2, h + h // 2), BLACK, -1, cv2.LINE_AA)
        cv2.putText(image, txt, (h // 4, h + h // 4), 0, fontscale, WHITE, fw, cv2.LINE_AA)
    return image


def colorize_mask(mask: np.ndarray, cmap: list[tuple[int, int, int]]) -> np.ndarray:
    colormap = [np.array(c, dtype=np.float32) for c in cmap]
    mask_shape = mask.shape
    new_mask = np.zeros((*mask_shape, 3))
    for i, c in enumerate(colormap):
        new_mask[mask == i] = c
    return new_mask.astype(np.uint8)
