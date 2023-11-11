"""Model summary callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.utils.files import save_txt_to_file


class ModelSummary(BaseCallback):
    def __init__(self, depth: int = 4):
        self.depth = depth

    def on_fit_start(self, trainer: Trainer):
        model = trainer.module.model
        model_summary = model.summary(self.depth)
        filepath = f"{trainer.logger.model_dir}/model_summary.txt"
        save_txt_to_file(model_summary, filepath)
