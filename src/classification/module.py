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

    def __init__(
        self,
        model: ClassificationModel,
        loss_fn: ClassificationLoss,
        optimizers: dict,
        lr_schedulers: dict,
        idx2label: dict[int, str],
    ):
        super().__init__(model, loss_fn, optimizers, lr_schedulers)
        self.idx2label = idx2label

    def batch_to_device(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        images, targets = batch
        return images.to(self.device), targets.to(self.device)

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

        results = []
        for i in range(len(images)):
            result = ClassificationResult(
                model_input_image=images[i],
                logits=logits.detach().cpu(),
                target_label=self.idx2label[int(targets[i].item())],
                idx2label=self.idx2label,
            )
            results.append(result)
        return metrics, []
