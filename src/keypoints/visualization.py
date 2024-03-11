from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.image import get_color, matplot_figure_to_array

sns.set_style("whitegrid")


def draw_elipsis(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    width: int | None = None,
):
    h, w = image.shape[:2]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    delta_x = x2 - x1
    delta_y = y2 - y1
    if width is None:
        img_diag = (h**2 + w**2) ** 0.5
        width = int(img_diag // 300)

    kpts_dist = int((delta_x**2 + delta_y**2) ** 0.5)
    if abs(delta_x) > abs(delta_y):
        a = kpts_dist // 2
        b = width
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    else:
        a = width
        b = kpts_dist // 2
        angle = np.arctan2(-delta_x, delta_y) * 180 / np.pi
    cv2.ellipse(image, (center_x, center_y), (a, b), angle, 0, 360, color, -1)


def plot_connections(
    image: np.ndarray,
    grouped_kpts_coords: np.ndarray,
    grouped_kpts_scores: np.ndarray,
    limbs: list[tuple[int, int]] | None = None,
    thr: float = 0.05,
    color_mode: Literal["person", "limb"] = "person",
    alpha: float = 0.8,
) -> np.ndarray:
    """
    grouped_kpts_coords is of shape [num_obj, num_kpts, 2]
    grouped_kpts_scores is of shape [num_obj, num_kpts, 1]
    """
    num_obj = len(grouped_kpts_coords)
    connections_image = image.copy()
    objs_sizes = grouped_kpts_coords[..., 1].max(1) - grouped_kpts_coords[..., 1].min(1)
    objs_draw_sizes = (objs_sizes / 100).astype(np.int32)
    for i in range(num_obj):
        obj_draw_size = max(2, objs_draw_sizes[i])
        obj_kpts_coords = grouped_kpts_coords[i]
        obj_kpts_scores = grouped_kpts_scores[i]
        if color_mode == "person":
            color = get_color(i).tolist()

        # draw limbs connections
        if limbs is not None:
            for j, (idx_0, idx_1) in enumerate(limbs):
                if obj_kpts_scores[idx_0] < thr or obj_kpts_scores[idx_1] < thr:
                    continue
                x1, y1 = obj_kpts_coords[idx_0]
                x2, y2 = obj_kpts_coords[idx_1]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)
                if color_mode == "limb":
                    color = get_color(j).tolist()
                draw_elipsis(connections_image, x1, y1, x2, y2, color, width=obj_draw_size)
                # cv2.line(connections_image, (x1, y1), (x2, y2), color, thickness)

        # draw keypoints
        for j, ((x, y), score) in enumerate(zip(obj_kpts_coords, obj_kpts_scores)):
            if score < thr:
                continue
            if color_mode == "limb":
                color = get_color(j).tolist()
            x, y = int(x), int(y)
            cv2.circle(connections_image, (x, y), obj_draw_size, color, -1)
            cv2.circle(connections_image, (x, y), obj_draw_size + 1, (0, 0, 0), 1)
    return cv2.addWeighted(image, 1 - alpha, connections_image, alpha, 0.0)


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


def plot_grouped_ae_tags(kpts_tags: np.ndarray) -> np.ndarray:
    num_obj, num_kpts, emb_dim = kpts_tags.shape

    fig, axes = plt.subplots(1, emb_dim, figsize=(6 * emb_dim, 6))
    if emb_dim == 1:
        axes = [axes]
    for e, ax in enumerate(axes):
        x = []
        y = []
        c = []
        for i in range(num_obj):
            for j in range(num_kpts):
                x.append(kpts_tags[i, j, e])
                y.append(j)
                c.append((get_color(i) / 255).tolist())
        ax.scatter(x, y, c=c, s=120)

    for i, ax in enumerate(axes):
        ax.set_yticks([i for i in range(num_kpts)])
        ax.set_xlabel("Embedding values", fontsize=12)
        if i == 0:
            ax.set_ylabel("Keypoint index", fontsize=12)
        ax.set_title(f"Embedding dim = {i}", fontsize=14)
    fig.suptitle("Associative Embeddings after grouping", fontsize=16)
    fig_data = matplot_figure_to_array(fig)
    plt.close()
    return fig_data
