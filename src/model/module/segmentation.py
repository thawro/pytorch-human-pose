"""Implementation of specialized Module"""
from .base import BaseModule
from torch import Tensor
from src.model.model.segmentation import SegmentationModel
from src.model.loss.segmentation import AuxiliarySegmentationLoss
from src.metrics.results import SegmentationResult
from src.model.metrics.segmentation import SegmentationMetrics


class SegmentationModule(BaseModule):
    model: SegmentationModel
    loss_fn: AuxiliarySegmentationLoss
    metrics: SegmentationMetrics
    results: dict[str, SegmentationResult]

    def _common_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        update_metrics: bool,
    ):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        images, masks, onehot_class_tensors = batch

        seg_preds, cls_preds = self.model.segment(images)

        loss = self.loss_fn.calculate_loss(
            seg_pred=seg_preds,
            seg_target=masks,
            cls_pred=cls_preds,
            cls_target=onehot_class_tensors,
        )

        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        if update_metrics:
            losses = {"aux_loss": loss.item()}
            metrics = self.metrics.calculate_metrics(seg_preds, masks)
            self.steps_metrics_storage.append(losses, self.stage)
            self.steps_metrics_storage.append(metrics, self.stage)

        if self.current_step % self.log_every_n_steps == 0 and batch_idx == 0:
            self.results[self.stage] = SegmentationResult(
                images=images.permute(0, 2, 3, 1).detach().cpu().numpy(),
                preds=seg_preds.detach().cpu().numpy(),
                targets=masks.cpu().numpy(),
            )
