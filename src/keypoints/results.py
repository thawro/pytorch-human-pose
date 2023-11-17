from dataclasses import dataclass
import numpy as np


@dataclass
class KeypointsResult:
    images: np.ndarray
    target_heatmaps: np.ndarray
    pred_heatmaps: np.ndarray
    keypoints: list[list[int]]
