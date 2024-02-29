"""Implementation of specialized Module"""

import torch
from torch import Tensor, optim

from src.base.lr_scheduler import LRScheduler
from src.base.module import BaseModule

from .datamodule import ClassificationDataModule
from .loss import ClassificationLoss
from .model import ClassificationModel


class BaseClassificationModule(BaseModule):
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


class ClassificationModule(BaseClassificationModule):
    model: ClassificationModel
    loss_fn: ClassificationLoss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        self.stage = "train"
        images, targets = batch
        logits = self.model(images)

        loss = self.loss_fn.calculate_loss(targets, logits)
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()
            if self.use_fp16:
                self.optimizers["optim"].backward(loss)
            else:
                loss.backward()
            self.optimizers["optim"].step()

        with torch.no_grad():
            logits = logits.detach().cpu()
            targets = targets.detach().cpu()
            loss = loss.detach().item()
            top_5 = logits.topk(k=5, dim=1).indices
            expanded_targets = targets.unsqueeze(-1).expand_as(top_5)
            top_5_acc = torch.any(top_5 == expanded_targets, dim=1).float().mean().item()
            top_1_acc = (top_5[:, 0] == targets).float().mean().item()
            metrics = {
                "loss": loss,
                "top-1_error": 1 - top_1_acc,
                "top-5_error": 1 - top_5_acc,
            }
            return metrics

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> dict[str, float]:
        self.stage = stage
        images, targets = batch
        logits = self.model(images)
        loss = self.loss_fn.calculate_loss(targets, logits)

        logits = logits.cpu()
        targets = targets.cpu()

        top_5 = logits.topk(k=5, dim=1).indices
        expanded_targets = targets.unsqueeze(-1).expand_as(top_5)
        top_5_acc = torch.any(top_5 == expanded_targets, dim=1).float().mean().item()
        top_1_acc = (top_5[:, 0] == targets).float().mean().item()
        metrics = {
            "loss": loss.item(),
            "top-1_error": 1 - top_1_acc,
            "top-5_error": 1 - top_5_acc,
        }
        return metrics, []
