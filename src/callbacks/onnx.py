"""Onnx models callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

import time
import math

from src.logging import get_pylogger
from .base import BaseCallback

log = get_pylogger(__name__)


class SaveLastAsOnnx(BaseCallback):
    def __init__(self, every_n_minutes: int = 30):
        self.every_n_minutes = every_n_minutes
        self.start_time = time.time()
        self.num_saved = 0

    def on_fit_start(self, trainer: Trainer):
        model = trainer.module.model
        dirpath = str(trainer.logger.model_onnx_dir)
        log.info("Saving model to onnx")
        filepath = f"{dirpath}/model.onnx"
        model.export_to_onnx(filepath)

    def on_validation_epoch_end(self, trainer: Trainer):
        model = trainer.module.model
        dirpath = str(trainer.logger.model_onnx_dir)
        filepath = f"{dirpath}/model.onnx"
        curr_time = time.time()
        diff_s = curr_time - self.start_time
        diff_min = math.ceil(diff_s / 60)
        if diff_min / self.every_n_minutes > 1 or self.num_saved == 0:
            self.start_time = curr_time
            log.info(
                f"{diff_min} minutes have passed. Saving model components to ONNX."
            )
            model.export_to_onnx(filepath)
            self.num_saved += 1
