"""Implementation of specialized Module"""
from torch import optim
from torch import Tensor

from src.base.module import BaseModule

from .model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from .loss import KeypointsLoss, AEKeypointsLoss
from .results import SPPEKeypointsResults, MPPEKeypointsResults
from .metrics import KeypointsMetrics


class BaseKeypointsModule(BaseModule):
    metrics: KeypointsMetrics

    def __init__(
        self,
        model: BaseKeypointsModel,
        loss_fn: KeypointsLoss,
        metrics: KeypointsMetrics,
        labels: list[str],
        optimizers: dict[str, optim.Optimizer],
        schedulers: dict[str, optim.lr_scheduler.LRScheduler] = {},
    ):
        super().__init__(model, loss_fn, metrics, optimizers, schedulers)
        self.labels = labels

    def _update_metrics(self, metrics: dict[str, float], losses: dict[str, float]):
        self.steps_metrics_storage.append(metrics, self.stage)
        self.current_epoch_steps_metrics_storage.append(metrics, self.stage)
        self.steps_metrics_storage.append(losses, self.stage)
        self.current_epoch_steps_metrics_storage.append(losses, self.stage)


class KeypointsModule(BaseKeypointsModule):
    model: KeypointsModel
    loss_fn: KeypointsLoss
    results: dict[str, SPPEKeypointsResults]

    def _common_step(
        self,
        batch: tuple[Tensor, list[Tensor], Tensor, Tensor, Tensor],
        batch_idx: int,
        update_metrics: bool,
    ):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        images, stages_target_heatmaps, target_weights, _, _ = batch

        stages_pred_heatmaps = self.model(images)

        loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps, stages_target_heatmaps, target_weights
        )
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        pred_heatmaps = stages_pred_heatmaps[-1]
        target_heatmaps = stages_target_heatmaps[-1]

        if update_metrics:
            losses = {"loss": loss.item()}
            metrics = self.metrics.calculate_metrics(pred_heatmaps, target_heatmaps)
            self._update_metrics(metrics, losses)

        if self.is_log_step(batch_idx):
            inv_processing = self.datamodule.transform.inverse_preprocessing
            numpy_images = inv_processing(images.detach().cpu().numpy())
            self.results[self.stage] = SPPEKeypointsResults.from_preds(
                numpy_images, pred_heatmaps
            )


class AEKeypointsModule(BaseKeypointsModule):
    model: AEKeypointsModel
    loss_fn: AEKeypointsLoss
    results: dict[str, MPPEKeypointsResults]

    def _common_step(
        self,
        batch: tuple[Tensor, list[Tensor], Tensor, list, list],
        batch_idx: int,
        update_metrics: bool,
    ):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        images, stages_target_heatmaps, target_weights, target_keypoints, _ = batch

        stages_pred_heatmaps, stages_pred_tags = self.model(images)

        loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps,
            stages_target_heatmaps,
            stages_pred_tags,
            target_weights,
            target_keypoints,
        )
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        pred_heatmaps = stages_pred_heatmaps[-1]
        pred_tags = stages_pred_tags[-1]
        target_heatmaps = stages_target_heatmaps[-1]

        if update_metrics:
            losses = {"loss": loss.item()}
            metrics = self.metrics.calculate_metrics(pred_heatmaps, target_heatmaps)
            self._update_metrics(metrics, losses)

        if self.is_log_step(batch_idx):
            inv_processing = self.datamodule.transform.inverse_preprocessing
            numpy_images = inv_processing(images.detach().cpu().numpy())
            self.results[self.stage] = MPPEKeypointsResults.from_preds(
                numpy_images, pred_heatmaps, pred_tags
            )
