from dataclasses import dataclass
import numpy as np


@dataclass
class Result:
    data: np.ndarray
    targets: np.ndarray
    preds: np.ndarray


@dataclass
class SegmentationResult:
    images: np.ndarray
    targets: np.ndarray
    preds: np.ndarray
