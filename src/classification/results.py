from dataclasses import dataclass, field
import numpy as np


@dataclass
class ClassificationResult:
    image: np.ndarray
    target: np.ndarray
    pred: np.ndarray
