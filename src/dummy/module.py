"""Implementation of specialized Module"""
from src.base.module import BaseModule
from torch import Tensor
from .model import DummyModel
from .loss import DummyLoss
from .results import DummyResult
from .metrics import DummyMetrics


class DummyModule(BaseModule):
    model: DummyModel
    loss_fn: DummyLoss
    metrics: DummyMetrics
    results: dict[str, DummyResult]

    def _common_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, update_metrics: bool
    ):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        data, targets = batch
        preds = self.model(data)
        loss = self.loss_fn.calculate_loss(preds, targets)

        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        if update_metrics:
            losses = {"loss": loss.item()}
            metrics = self.metrics.calculate_metrics(preds, targets)
            self.steps_metrics_storage.append(metrics, self.stage)
            self.steps_metrics_storage.append(losses, self.stage)

            self.current_epoch_steps_metrics_storage.append(losses, self.stage)
            self.current_epoch_steps_metrics_storage.append(metrics, self.stage)

        if self.current_step % self.log_every_n_steps == 0 and batch_idx == 0:
            self.results[self.stage] = DummyResult(
                data=data.detach().cpu().numpy(),
                preds=preds.detach().cpu().numpy(),
                targets=targets.cpu().numpy(),
            )
