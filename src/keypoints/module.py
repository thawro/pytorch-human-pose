"""Implementation of specialized Module"""
from torch import optim, Tensor
from typing import Type

from src.base.module import BaseModule

from .model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from .loss import KeypointsLoss, AEKeypointsLoss
from .datamodule import KeypointsDataModule
from .results import MPPEKeypointsResult
from src.base.lr_scheduler import LRScheduler
import torch.nn.functional as F
from src.utils.fp16_utils.fp16_optimizer import FP16_Optimizer


_batch = tuple[Tensor, list[Tensor], Tensor, list, list]


class BaseKeypointsModule(BaseModule):
    datamodule: KeypointsDataModule

    def __init__(
        self,
        model: BaseKeypointsModel,
        loss_fn: KeypointsLoss,
        labels: list[str],
        limbs: list[tuple[int, int]],
    ):
        super().__init__(model, loss_fn)
        self.labels = labels
        self.limbs = limbs

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        fp16_enabled = True
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # if fp16_enabled:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, verbose=False)

        optimizers = {"optim": optimizer}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    optimizers["optim"].optimizer,
                    milestones=[130, 170, 200],
                    gamma=0.1,
                ),
                interval="epoch",
            )
        }
        return optimizers, schedulers


class SPPEKeypointsModule(BaseKeypointsModule):
    model: KeypointsModel
    loss_fn: KeypointsLoss

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        (
            images,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            target_visibilities,
        ) = batch

        stages_pred_heatmaps = self.model(images)

        loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps, stages_target_heatmaps, target_weights
        )
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        metrics = {"loss": loss.item()}

        return metrics


class MPPEKeypointsModule(BaseKeypointsModule):
    model: AEKeypointsModel
    loss_fn: AEKeypointsLoss

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        (
            images,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            target_visibilities,
        ) = batch
        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.model(images)

        hm_loss, tags_loss = self.loss_fn.calculate_loss(
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            target_visibilities,
        )
        loss = hm_loss + tags_loss
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()
            # if fp16:
            self.optimizers["optim"].backward(loss)
            # else:
            # loss.backward()

            self.optimizers["optim"].step()

        metrics = {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "tags_loss": tags_loss.item(),
        }

        if self.stage == "train":
            return metrics

        results = []
        for i in range(len(images)):
            _image = self.datamodule.transform.inverse_preprocessing(images[i].detach())
            pred_kpts_heatmaps = [hms[i].detach() for hms in stages_pred_kpts_heatmaps]
            pred_tags_heatmaps = [hms[i].detach() for hms in stages_pred_tags_heatmaps]

            result = MPPEKeypointsResult(
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
