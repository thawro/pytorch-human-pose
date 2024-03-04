"""Implementation of specialized Module"""

import torch
from torch import Tensor, optim

from src.base.lr_scheduler import LRScheduler
from src.base.module import BaseModule

from .datamodule import ClassificationDataModule
from .loss import ClassificationLoss
from .model import ClassificationModel
from .results import ClassificationResult


def get_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()
    top_5 = logits.topk(k=5, dim=1).indices
    expanded_targets = targets.unsqueeze(-1).expand_as(top_5)
    top_5_acc = torch.any(top_5 == expanded_targets, dim=1).float().mean().item()
    top_1_acc = (top_5[:, 0] == targets).float().mean().item()
    return {"top-1_error": 1 - top_1_acc, "top-5_error": 1 - top_5_acc}


class ClassificationModule(BaseModule):
    model: ClassificationModel
    loss_fn: ClassificationLoss
    datamodule: ClassificationDataModule

    def __init__(self, model: ClassificationModel, loss_fn: ClassificationLoss, labels: list[str]):
        super().__init__(model, loss_fn)
        self.labels = labels

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.net.parameters()),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True,
        )

        optimizers = {"optim": optimizer}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[30, 60, 90],
                    gamma=0.1,
                ),
                interval="epoch",
            )
        }
        return optimizers, schedulers

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        images, targets = batch
        logits = self.model.net(images)

        loss = self.loss_fn.calculate_loss(targets, logits)
        self.optimizers["optim"].zero_grad()
        loss.backward()
        self.optimizers["optim"].step()

        with torch.no_grad():
            metrics = get_metrics(logits, targets)
            metrics["loss"] = loss = loss.detach().item()
        return metrics

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> tuple[dict[str, float], list[ClassificationResult]]:
        images, targets = batch
        logits = self.model.net(images)

        loss = self.loss_fn.calculate_loss(targets, logits)

        metrics = get_metrics(logits, targets)
        metrics["loss"] = loss = loss.detach().item()
        return metrics, []
