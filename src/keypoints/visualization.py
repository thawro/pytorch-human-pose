import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.image import make_grid, get_color
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import SPPEKeypointsResults, MPPEKeypointsResults


def plot_connections(
    image: np.ndarray,
    all_kpts_coords: np.ndarray,
    all_kpts_scores: np.ndarray,
    limbs: list[tuple[int, int]] | None,
    thr: float = 0.05,
):
    """
    all_kpts_coords is of shape [num_obj, num_kpts, 2]
    all_kpts_scores is of shape [num_obj, num_kpts, 1]

    """
    h, w = image.shape[:2]
    radius = max(h, w) // 100 + -2
    thickness = max(h, w) // 100

    for i in range(len(all_kpts_coords)):
        kpts_coords = all_kpts_coords[i]
        kpts_scores = all_kpts_scores[i]

        color = get_color(i).tolist()

        if limbs is not None:
            for id_1, id_2 in limbs:
                if kpts_scores[id_1] < thr or kpts_scores[id_2] < thr:
                    continue
                x1, y1 = kpts_coords[id_1]
                x2, y2 = kpts_coords[id_2]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        for (x, y), score in zip(kpts_coords, kpts_scores):
            if score < thr:
                continue
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)
    return image


def plot_heatmaps(
    image: np.ndarray,
    heatmaps: np.ndarray,
    clip_0_1: bool = False,
    minmax: bool = False,
) -> list[np.ndarray]:
    heatmaps_vis = []
    for hm in heatmaps:
        if clip_0_1:
            hm = np.clip(hm, 0, 1)
        if minmax:
            hm = (hm - hm.max()) / (hm.max() - hm.min())
        hm = (hm * 255).astype(np.uint8)
        hm = 255 - hm
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        img_hm = cv2.addWeighted(image, 0.25, hm, 0.75, 0)
        heatmaps_vis.append(img_hm)
    return heatmaps_vis


def plot_sppe_results_heatmaps(
    results: "SPPEKeypointsResults",
    limbs: list[tuple[int, int]],
    filepath: str | None = None,
    thr: float = 0.2,
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

        pred_kpts_heatmaps = plot_heatmaps(
            image, pred_heatmaps, clip_0_1=True, minmax=False
        )

        image = plot_connections(image.copy(), kpts_coords, kpts_scores, limbs, thr)
        pred_kpts_heatmaps.insert(0, image)

        pred_hms_grid = make_grid(pred_kpts_heatmaps, nrows=2, pad=5)
        grids.append(pred_hms_grid)
        ax.imshow(pred_hms_grid)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()
    return grids


def plot_mppe_results_heatmaps(
    results: "MPPEKeypointsResults",
    limbs: list[tuple[int, int]],
    filepath: str | None = None,
    thr: float = 0.05,
):
    n_rows = min(10, len(results.pred_heatmaps))
    fig, axes = plt.subplots(n_rows, 1, figsize=(24, n_rows * 16))
    grids = []
    for i in range(n_rows):
        ax = axes[i]
        pred_kpts_heatmaps = results.pred_heatmaps[i]
        pred_tags_heatmaps = results.pred_tags[i]

        image = results.images[i]
        pred_keypoints = results.pred_keypoints[i].astype(np.int32)
        pred_scores = results.pred_scores[i]

        image = plot_connections(
            image.copy(),
            pred_keypoints,
            pred_scores,
            limbs,
            thr=thr,
        )

        final_plots = []
        num_stages = 2
        for i in range(num_stages):
            kpts_heatmaps_plots = plot_heatmaps(
                image, pred_kpts_heatmaps[..., i], clip_0_1=True, minmax=False
            )
            tags_heatmaps_plots = plot_heatmaps(
                image, pred_tags_heatmaps[..., i], clip_0_1=False, minmax=True
            )
            kpts_heatmaps_plots.insert(0, image)
            tags_heatmaps_plots.insert(0, image)

            kpts_grid = make_grid(kpts_heatmaps_plots, nrows=2, pad=5)
            tags_grid = make_grid(tags_heatmaps_plots, nrows=2, pad=5)
            final_plots.extend([kpts_grid, tags_grid])

        final_plot = np.concatenate(final_plots, axis=0)
        final_plot = cv2.resize(final_plot, dsize=(0, 0), fx=0.4, fy=0.4)

        grids.append(final_plot)
        ax.imshow(final_plot)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()
    return grids
