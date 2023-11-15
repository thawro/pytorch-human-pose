"""Training metrics callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.utils.files import save_yaml
from src.visualization import plot_metrics
from src.logging import get_pylogger

log = get_pylogger(__name__)


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def plot(self, trainer: Trainer, on_step: bool) -> None:
        module = trainer.module
        prefix = "step" if on_step else "epoch"
        filepath = f"{trainer.logger.log_path}/{prefix}_metrics.jpg"
        storage = (
            module.steps_metrics_storage if on_step else module.epochs_metrics_storage
        )
        if len(storage.metrics) > 0:
            plot_metrics(storage, filepath=filepath)
        else:
            log.warn("No metrics to plot logged yet")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, on_step=False)

    def log(self, trainer: Trainer) -> None:
        self.plot(trainer, on_step=True)


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def save(self, trainer: Trainer, on_step: bool) -> None:
        module = trainer.module
        prefix = "step" if on_step else "epoch"
        filepath = filepath = f"{trainer.logger.log_path}/{prefix}_metrics.yaml"

        storage = (
            module.steps_metrics_storage if on_step else module.epochs_metrics_storage
        )
        if len(storage.metrics) > 0:
            metrics = storage.to_dict()
            save_yaml(metrics, filepath)
        else:
            log.warn("No metrics to save logged yet")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.save(trainer, on_step=False)

    def log(self, trainer: Trainer) -> None:
        self.save(trainer, on_step=True)
