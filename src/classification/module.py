"""Implementation of specialized Module"""

from torch import optim, Tensor
from typing import Type

from src.base.module import BaseModule

from .model import ClassificationModel
from .loss import ClassificationLoss
from .datamodule import ClassificationDataModule
from .results import ClassificationResult
from src.base.lr_scheduler import LRScheduler
import torch.nn.functional as F
from src.utils.fp16_utils.fp16_optimizer import FP16_Optimizer


_batch = tuple[Tensor, Tensor]


class BaseClassificationModule(BaseModule):
    datamodule: ClassificationDataModule

    def __init__(
        self,
        model: ClassificationModel,
        loss_fn: ClassificationLoss,
        labels: list[str],
    ):
        super().__init__(model, loss_fn)
        self.labels = labels

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        fp16_enabled = True
        optimizer = optim.Adam(self.model.parameters(), lr=1e-1)
        # if fp16_enabled:
        # TODO: make it configurable (along with distributed)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, verbose=False)

        optimizers = {"optim": optimizer}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    optimizers["optim"].optimizer,
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

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, targets = batch
        logits = self.model(images)

        loss = self.loss_fn.calculate_loss(targets, logits)
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()
            # if fp16:
            self.optimizers["optim"].backward(loss)
            # else:
            # loss.backward()
            self.optimizers["optim"].step()

        metrics = {"loss": loss.item()}

        if self.stage == "train":
            return metrics

        results = []
        for i in range(len(images)):
            _image = self.datamodule.transform.inverse_preprocessing(images[i].detach())

            result = ClassificationResult(
                image=_image,
                stages_pred_kpts_heatmaps=pred_kpts_heatmaps,
                stages_pred_tags_heatmaps=pred_tags_heatmaps,
                limbs=self.limbs,
                max_num_people=20,
                det_thr=0.1,
                tag_thr=1.0,
            )
            results.append(result)
        return metrics, results
