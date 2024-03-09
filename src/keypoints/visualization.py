import cv2
import numpy as np

from src.utils.image import get_color


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
