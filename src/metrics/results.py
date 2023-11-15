from dataclasses import dataclass
import numpy as np


@dataclass
class Result:
    data: np.ndarray
    targets: np.ndarray
    preds: np.ndarray


@dataclass
class SegmentationResult:
    image: np.ndarray
    target: np.ndarray
    pred: np.ndarray


@dataclass
class KeypointsResult:
    images: np.ndarray
    target_heatmaps: np.ndarray
    pred_heatmaps: np.ndarray
    keypoints: list[list[int]]
