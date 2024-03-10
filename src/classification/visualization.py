import numpy as np

from src.utils.image import put_txt


def plot_top_preds(
    image: np.ndarray,
    probs: np.ndarray,
    idx2label: dict[int, str],
    k: int = 5,
    target_label: str | int | None = None,
) -> np.ndarray:
    min_size = min(image.shape[:2])
    if min_size > 0 and min_size <= 224:
        font_scale = 0.3
    elif min_size > 224 and min_size <= 336:
        font_scale = 0.5
    else:
        font_scale = 0.7
    top_idxs = np.argsort(probs)[-k:]
    top_idxs = list(reversed(top_idxs))
    labels = [f"{probs[idx]:.3f}:  {idx2label[idx]}" for idx in top_idxs]
    image_plot = put_txt(image.copy(), labels, alpha=0.8, font_scale=font_scale, thickness=1)

    if target_label is not None:
        image_plot = put_txt(
            image_plot,
            ["Target:", str(target_label)],
            loc="br",
            txt_color=(80, 255, 80),
            alpha=0.8,
            font_scale=font_scale,
        )
    return image_plot
