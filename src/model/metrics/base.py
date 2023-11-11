from torch import Tensor
from abc import abstractmethod


class BaseMetrics:
    @abstractmethod
    def calculate_metrics(self, y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
        raise NotImplementedError()
