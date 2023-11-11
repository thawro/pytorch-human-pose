from torch import Tensor
from src.metrics.confusionmatrix import ConfusionMatrix
from .base import BaseMetrics


class SegmentationMetrics(BaseMetrics):
    def __init__(self, num_classes: int):
        self.cm = ConfusionMatrix(num_classes)

    def calculate_metrics(self, y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
        self.cm.add(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy())
        return self.cm.compute()
