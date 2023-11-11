from src.model.metrics.base import BaseMetrics
from torch import Tensor


class DummyMetrics(BaseMetrics):
    def calculate_metrics(self, y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
        mae = (y_pred - y_true).abs().mean()
        return {"MAE": mae.item()}
