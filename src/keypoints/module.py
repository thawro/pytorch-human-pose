"""Implementation of specialized Module"""

import torch
from torch import Tensor, optim

from src.base.lr_scheduler import LRScheduler
from src.base.module import BaseModule

from .datamodule import KeypointsDataModule
from .loss import AEKeypointsLoss
from .model import AEKeypointsModel, BaseKeypointsModel, KeypointsModel
from .results import MPPEKeypointsResult

_batch = tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]


class BaseKeypointsModule(BaseModule):
    datamodule: KeypointsDataModule

    def __init__(
        self,
        model: BaseKeypointsModel,
        loss_fn: AEKeypointsLoss,
        labels: list[str],
        limbs: list[tuple[int, int]],
    ):
        super().__init__(model, loss_fn)
        self.labels = labels
        self.limbs = limbs

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        optimizers = {"optim": optimizer}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[130, 170, 200],
                    gamma=0.1,
                ),
                interval="epoch",
            )
        }
        return optimizers, schedulers


class MPPEKeypointsModule(BaseKeypointsModule):
    model: AEKeypointsModel
    loss_fn: AEKeypointsLoss

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, heatmaps, masks, joints = batch

        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        images = images.cuda()

        scaler, optimizer = self.scalers["optim"], self.optimizers["optim"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            stages_pred_kpts_heatmaps, pred_tags_heatmaps = self.model(images)
            hm_loss, tags_loss = self.loss_fn.calculate_loss(
                stages_pred_kpts_heatmaps, pred_tags_heatmaps, heatmaps, masks, joints
            )
            loss = hm_loss + tags_loss

        if self.stage == "train":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metrics = {
            "loss": loss.detach().item(),
            "hm_loss": hm_loss.detach().item(),
            "tags_loss": tags_loss.detach().item(),
        }

        if self.stage == "train":
            return metrics

        pred_tags_heatmaps = pred_tags_heatmaps.detach()
        stages_pred_kpts_heatmaps = [hms.detach() for hms in stages_pred_kpts_heatmaps]
        images = images.detach().cpu()
        results = []
        for i in range(len(images)):
            result = MPPEKeypointsResult(
                image=images[i],
                stages_pred_kpts_heatmaps=[hms[i] for hms in stages_pred_kpts_heatmaps],
                tags_heatmaps=pred_tags_heatmaps[i],
                limbs=self.limbs,
                max_num_people=20,
                det_thr=0.1,
                tag_thr=1.0,
            )
            results.append(result)
        return metrics, results
