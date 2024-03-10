"""Implementation of specialized Module"""

import torch
from torch import Tensor

from src.base.module import BaseModule
from src.logger.loggers import log

from .datamodule import KeypointsDataModule
from .loss import AEKeypointsLoss
from .model import KeypointsModel
from .results import KeypointsResult

_batch = tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]


class KeypointsModule(BaseModule):
    model: KeypointsModel
    loss_fn: AEKeypointsLoss
    datamodule: KeypointsDataModule

    def __init__(
        self,
        model: KeypointsModel,
        loss_fn: AEKeypointsLoss,
        labels: list[str],
        limbs: list[tuple[int, int]],
        optimizers: dict,
        lr_schedulers: dict,
    ):
        super().__init__(model, loss_fn, optimizers, lr_schedulers)
        self.labels = labels
        self.limbs = limbs
        log.info("..Using torch autocast to float16 in modules forward implementation..")

    def batch_to_device(self, batch: _batch) -> _batch:
        images, heatmaps, masks, joints = batch
        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        images = images.cuda()
        return images, heatmaps, masks, joints

    def training_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, heatmaps, masks, joints = batch

        scaler, optimizer = self.scalers["optim"], self.optimizers["optim"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            stages_pred_kpts_heatmaps, pred_tags_heatmaps = self.model.net(images)
            heatmap_losses, push_losses, pull_losses = self.loss_fn.calculate_loss(
                stages_pred_kpts_heatmaps, pred_tags_heatmaps, heatmaps, masks, joints
            )
            loss = 0
            for i in range(len(heatmap_losses)):
                loss = loss + heatmap_losses[i]
            loss = loss + push_losses[0]
            loss = loss + pull_losses[0]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = {"loss": loss.detach().item()}
        for i, hm_loss in enumerate(heatmap_losses):
            metrics[f"hm_{i}_loss"] = hm_loss.item()
        for i in range(len(push_losses)):
            metrics[f"push_{i}_loss"] = push_losses[i].item()
            metrics[f"pull_{i}_loss"] = pull_losses[i].item()

        return metrics

    def validation_step(
        self, batch: _batch, batch_idx: int
    ) -> tuple[dict[str, float], list[KeypointsResult]]:
        images, heatmaps, masks, joints = batch

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            stages_pred_kpts_heatmaps, pred_tags_heatmaps = self.model.net(images)
            heatmap_losses, push_losses, pull_losses = self.loss_fn.calculate_loss(
                stages_pred_kpts_heatmaps, pred_tags_heatmaps, heatmaps, masks, joints
            )
            loss = 0
            for i in range(2):
                loss = loss + heatmap_losses[i]
            loss = loss + push_losses[0]
            loss = loss + pull_losses[0]

        metrics = {"loss": loss.detach().item()}
        for i, hm_loss in enumerate(heatmap_losses):
            metrics[f"hm_{i}_loss"] = hm_loss.item()
        for i in range(len(push_losses)):
            metrics[f"push_{i}_loss"] = push_losses[i].item()
            metrics[f"pull_{i}_loss"] = pull_losses[i].item()

        pred_tags_heatmaps = pred_tags_heatmaps.detach()
        stages_pred_kpts_heatmaps = [hms.detach() for hms in stages_pred_kpts_heatmaps]
        images = images.detach().cpu()
        results = []
        for i in range(len(images)):
            result = KeypointsResult(
                model_input_image=images[i],
                kpts_heatmaps=[hms[i].unsqueeze(0) for hms in stages_pred_kpts_heatmaps],
                tags_heatmaps=pred_tags_heatmaps[i].unsqueeze(0),
                limbs=self.limbs,
                max_num_people=20,
                det_thr=0.1,
                tag_thr=1.0,
            )
            results.append(result)
        return metrics, results
