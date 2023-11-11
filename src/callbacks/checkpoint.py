from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

import torch

from src.logging import get_pylogger
from .base import BaseCallback

log = get_pylogger(__name__)


class SaveModelCheckpoint(BaseCallback):
    def __init__(
        self,
        name: str | None = None,
        stage: str | None = None,
        metric: str | None = None,
        mode: str | None = "min",
        last: bool = False,
        top_k: int = 1,
        verbose: bool = False,
    ):
        self.name = name if name is not None else "best"
        self.stage = stage
        self.metric = metric
        self.save_last = last
        self.top_k = top_k
        self.verbose = verbose

        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def on_validation_epoch_end(self, trainer: Trainer):
        ckpt_dir = trainer.logger.ckpt_dir
        if self.metric is not None and self.stage is not None:
            metrics_storage = trainer.module.epochs_metrics_storage
            stage_metric_values = metrics_storage.get(self.metric, self.stage)
            if len(stage_metric_values) == 0:
                raise ValueError(
                    f"{self.metric} not yet logged to metrics storage. Current logged metrics: {metrics_storage.logged_metrics}"
                )
            last = stage_metric_values[-1]
            if self.compare(last, self.best) and self.top_k == 1:
                self.best = last
                log.info(f"Found new best value for {self.metric} ({self.stage})")
                trainer.save_checkpoint(str(ckpt_dir / f"{self.name}.pt"))
        if self.save_last:
            name = self.name if self.name is not None else "last"
            trainer.save_checkpoint(str(ckpt_dir / f"{name}.pt"))

    def state_dict(self) -> dict:
        return {f"best_{self.stage}_{self.metric}": self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict[f"best_{self.stage}_{self.metric}"]


class LoadModelCheckpoint(BaseCallback):
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    def on_fit_start(self, trainer: Trainer):
        trainer.load_checkpoint(self.ckpt_path)
