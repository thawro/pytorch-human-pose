import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image import make_grid
import random
from .results import SPPEKeypointsResults


def plot_connections(
    image: np.ndarray,
    all_kpts_coords: np.ndarray,
    all_kpts_scores: np.ndarray,
    limbs: list[tuple[int, int]],
    thr: float = -100,
):
    """
    all_kpts_coords is of shape [num_obj, num_kpts, 2]
    all_kpts_scores is of shape [num_obj, num_kpts, 1]

    """
    for i in range(len(all_kpts_coords)):
        kpts_coords = all_kpts_coords[i]
        kpts_scores = all_kpts_scores[i]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for (x, y), score in zip(kpts_coords, kpts_scores):
            if score < thr:
                continue
            cv2.circle(image, (x, y), 3, (0, 128, 255), -1)

        for id_1, id_2 in limbs:
            if kpts_scores[id_1] < thr or kpts_scores[id_2] < thr:
                continue
            x1, y1 = kpts_coords[id_1]

            x2, y2 = kpts_coords[id_2]
            cv2.line(image, (x1, y1), (x2, y2), color, 4)
    return image


def plot_heatmaps(image: np.ndarray, heatmaps: np.ndarray) -> list[np.ndarray]:
    kpts_heatmaps = []
    for hm in heatmaps:
        hm = np.clip(hm, 0, 1)
        hm = (hm * 255).astype(np.uint8)
        hm = 255 - hm
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        img_hm = cv2.addWeighted(image, 0.5, hm, 0.5, 1)
        kpts_heatmaps.append(img_hm)
    return kpts_heatmaps


def plot_results_heatmaps(
    results: SPPEKeypointsResults,
    limbs: list[tuple[int, int]],
    filepath: str | None = None,
    thr: float = -100,
):
    n_rows = min(10, len(results.pred_heatmaps))
    fig, axes = plt.subplots(n_rows, 1, figsize=(24, n_rows * 8))
    grids = []
    for i in range(n_rows):
        ax = axes[i]
        pred_heatmaps = results.pred_heatmaps[i]
        image = results.images[i]
        kpts_coords = results.pred_keypoints[i].astype(np.int32)
        kpts_scores = results.pred_scores[i]

        pred_kpts_heatmaps = plot_heatmaps(image, pred_heatmaps)

        image = plot_connections(image.copy(), kpts_coords, kpts_scores, limbs, thr)
        pred_kpts_heatmaps.insert(0, image)

        pred_hms_grid = make_grid(pred_kpts_heatmaps, nrows=2, pad=5)
        grids.append(pred_hms_grid)
        ax.imshow(pred_hms_grid)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    return grids
