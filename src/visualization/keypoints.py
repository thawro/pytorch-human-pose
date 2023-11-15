import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.metrics.results import KeypointsResult


def make_grid(images: list[np.ndarray], nrows: int = 2, pad: int = 5) -> np.ndarray:
    h, w = images[0].shape[:2]
    ncols = int(np.ceil(len(images) / nrows).item())
    grid_h = (h + pad) * nrows + pad
    grid_w = (w + pad) * ncols + pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=images[0].dtype)
    for idx, image in enumerate(images):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        row = idx // ncols
        col = idx % ncols
        grid_y = pad + row * (h + pad)
        grid_x = pad + col * (w + pad)
        grid[grid_y : grid_y + h, grid_x : grid_x + w] = image
    return grid


def plot_single_image(image: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    images = [image]
    for kpt_idx, heatmap in enumerate(heatmaps):
        heatmap = cv2.resize(heatmap, dsize=(h, w))
        hm = heatmap * 255
        hm = np.clip(hm, 0, 255).astype(np.uint8)
        hm = 255 - hm
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        img_hm = cv2.addWeighted(image, 0.5, hm, 0.5, 1)
        images.append(img_hm)

    hms_grid = make_grid(images, nrows=2, pad=5)
    return hms_grid


def plot_heatmaps(results: KeypointsResult, filepath: str | None = None):
    pred_heatmaps = results.pred_heatmaps[0]
    target_heatmaps = results.target_heatmaps[0]
    image = results.images[0]

    pred_hms_grid = plot_single_image(image, pred_heatmaps)
    target_hms_grid = plot_single_image(image, target_heatmaps)

    fig, axes = plt.subplots(2, 1, figsize=(24, 24))
    axes[0].imshow(pred_hms_grid)
    axes[1].imshow(target_hms_grid)

    if filepath is not None:
        fig.savefig(filepath)
