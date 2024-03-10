import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.image import get_color, matplot_figure_to_array

sns.set_style("whitegrid")


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
    radius = max(2, max(h, w) // 200 - 4)
    thickness = max(2, max(h, w) // 200 - 4)
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
    return matplot_figure_to_array(fig)
