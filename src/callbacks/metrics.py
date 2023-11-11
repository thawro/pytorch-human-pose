"""Training metrics callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.utils.files import save_yaml
from src.visualization import plot_metrics


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def on_epoch_end(self, trainer: Trainer) -> None:
        filepath = f"{trainer.logger.log_path}/metrics.jpg"
        metrics_storage = trainer.module.epochs_metrics_storage
        plot_metrics(metrics_storage, filepath=filepath)


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def on_epoch_end(self, trainer: Trainer) -> None:
        filepath = f"{trainer.logger.log_path}/metrics.yaml"
        metrics_storage = trainer.module.epochs_metrics_storage
        metrics = metrics_storage.to_dict()
        save_yaml(metrics, filepath)
