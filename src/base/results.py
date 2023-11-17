from dataclasses import dataclass
import numpy as np


@dataclass
class BaseResult:
    data: np.ndarray
    targets: np.ndarray
    preds: np.ndarray
