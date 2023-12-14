import numpy as np
import random
import cv2


def plot_connections(
    image: np.ndarray,
    all_kpts_coords: np.ndarray,
    all_kpts_scores: np.ndarray,
    limbs: list[tuple[int, int]],
    thr: float = 0.2,
):
    """
    all_kpts_coords is of shape [num_obj, num_kpts, 2]
    all_kpts_scores is of shape [num_obj, num_kpts, 1]

    """
    for i in range(len(all_kpts_coords)):
        kpts_coords = all_kpts_coords[i]
        kpts_scores = all_kpts_scores[i]
        color = (50, 200, 100)
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


def make_grid(
    images: list[np.ndarray], nrows: int = 2, pad: int = 5, resize: float = -1
) -> np.ndarray:
    _images = [img.copy() for img in images]
    if resize > 0:
        for i in range(len(_images)):
            _images[i] = cv2.resize(_images[i], (0, 0), fx=resize, fy=resize)

    h, w = _images[0].shape[:2]
    ncols = int(np.ceil(len(_images) / nrows).item())
    grid_h = (h + pad) * nrows + pad
    grid_w = (w + pad) * ncols + pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=_images[0].dtype)
    for idx, image in enumerate(_images):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        row = idx // ncols
        col = idx % ncols
        grid_y = pad + row * (h + pad)
        grid_x = pad + col * (w + pad)
        grid[grid_y : grid_y + h, grid_x : grid_x + w] = image
    return grid


def visualize(
    image: np.ndarray,
    heatmaps: np.ndarray,
    all_kpts_coords: np.ndarray,
    all_kpts_scores: np.ndarray,
    limbs: list,
    thr: float = 0.2,
):
    kpts_heatmaps = plot_heatmaps(image, heatmaps)

    image = plot_connections(image.copy(), all_kpts_coords, all_kpts_scores, limbs, thr)
    kpts_heatmaps.insert(0, image)

    hms_grid = make_grid(kpts_heatmaps, nrows=2, pad=5)
    return hms_grid
