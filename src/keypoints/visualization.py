import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image import make_grid

from .results import KeypointsResult


def plot_connections(
    image: np.ndarray,
    heatmaps: np.ndarray,
    limbs: list[tuple[int, int]],
    thr: float = 0.5,
):
    kpts = [np.unravel_index(np.argmax(hm), hm.shape) for hm in heatmaps]
    scores = [hm[kpt] for hm, kpt in zip(heatmaps, kpts)]

    for (y, x), score in zip(kpts, scores):
        if score < thr:
            continue
        cv2.circle(image, (x, y), 3, (0, 128, 255), -1)

    for id_1, id_2 in limbs:
        if scores[id_1] < thr or scores[id_2] < thr:
            continue
        kpt_1 = kpts[id_1]
        y1, x1 = kpt_1

        kpt_2 = kpts[id_2]
        y2, x2 = kpt_2
        cv2.line(image, (x1, y1), (x2, y2), (50, 255, 50), 4)
    return image


def create_heatmaps(
    image: np.ndarray,
    heatmaps: np.ndarray,
    limbs: list[tuple[int, int]],
    thr: float = 0.5,
) -> list[np.ndarray]:
    h, w = image.shape[:2]
    resized_heatmaps = []
    for i, hm in enumerate(heatmaps):
        resized_heatmaps.append(cv2.resize(hm, dsize=(w, h)))
    heatmaps = np.stack(resized_heatmaps)

    kpts_heatmaps = []
    for hm in heatmaps:
        hm = np.clip(hm, 0, 1)
        hm = (hm * 255).astype(np.uint8)
        hm = 255 - hm
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        img_hm = cv2.addWeighted(image, 0.5, hm, 0.5, 1)
        kpts_heatmaps.append(img_hm)
    image = plot_connections(image.copy(), heatmaps, limbs, thr)
    kpts_heatmaps.insert(0, image)
    return kpts_heatmaps


def plot_heatmaps(
    results: KeypointsResult,
    limbs: list[tuple[int, int]],
    filepath: str | None = None,
    thr: float = 0.5,
):
    n_rows = len(results.pred_heatmaps)
    fig, axes = plt.subplots(n_rows, 1, figsize=(24, n_rows * 8))
    grids = []
    for i in range(n_rows):
        ax = axes[i]
        pred_heatmaps = results.pred_heatmaps[i]
        image = results.images[i]
        pred_kpts_heatmaps = create_heatmaps(image, pred_heatmaps, limbs, thr)
        pred_kpts_heatmaps.insert(0, image)
        pred_hms_grid = make_grid(pred_kpts_heatmaps, nrows=2, pad=5)
        grids.append(pred_hms_grid)
        ax.imshow(pred_hms_grid)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    return grids
