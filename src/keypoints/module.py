"""Implementation of specialized Module"""
from torch import optim
from torch import Tensor
from typing import Type

from src.base.module import BaseModule
import torchvision.transforms.functional as F

from .model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from .loss import KeypointsLoss, AEKeypointsLoss
from .results import SPPEKeypointsResults, MPPEKeypointsResults
from .datamodule import KeypointsDataModule

_batch = tuple[Tensor, list[Tensor], Tensor, list, list, list]


class BaseKeypointsModule(BaseModule):
    datamodule: KeypointsDataModule

    def __init__(
        self,
        model: BaseKeypointsModel,
        loss_fn: KeypointsLoss,
        labels: list[str],
        optimizers: dict[str, optim.Optimizer],
        schedulers: dict[str, optim.lr_scheduler.LRScheduler],
        ResultsClass: Type[SPPEKeypointsResults | MPPEKeypointsResults],
    ):
        super().__init__(model, loss_fn, optimizers, schedulers)
        self.labels = labels
        self.ResultsClass = ResultsClass

    def _update_metrics(self, metrics: dict[str, float]):
        self.steps_metrics.append(
            metrics, self.current_step, self.current_epoch, self.stage
        )


class SPPEKeypointsModule(BaseKeypointsModule):
    model: KeypointsModel
    loss_fn: KeypointsLoss
    results: dict[str, SPPEKeypointsResults]
    ResultsClass: Type[SPPEKeypointsResults]

    def _common_step(self, batch: _batch, batch_idx: int):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        (
            images,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            target_visibilities,
            extra_coords,
        ) = batch

        stages_pred_heatmaps = self.model(images)

        loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps, stages_target_heatmaps, target_weights
        )
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        pred_heatmaps = stages_pred_heatmaps[-1]
        target_heatmaps = stages_target_heatmaps[-1]

        losses = {"loss": loss.item()}

        metrics_params = dict(
            step=self.current_step, epoch=self.current_epoch, split=self.stage
        )
        self.steps_metrics.append(losses, **metrics_params)

        if self.is_log_step and "eval" in self.stage:
            inv_processing = self.datamodule.transform.inverse_preprocessing
            numpy_images = inv_processing(images.detach().cpu().numpy())
            results = self.ResultsClass.from_preds(
                numpy_images,
                target_heatmaps,
                pred_heatmaps.detach(),
                target_keypoints,
                extra_coords,
                det_thr=0.2,
            )
            self.results[self.stage] = results
            metrics = results.evaluate()
            self.validation_metrics.append(metrics | losses, **metrics_params)


class MPPEKeypointsModule(BaseKeypointsModule):
    model: AEKeypointsModel
    loss_fn: AEKeypointsLoss
    results: dict[str, MPPEKeypointsResults]
    ResultsClass: Type[MPPEKeypointsResults]

    def _common_step(self, batch: _batch, batch_idx: int):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        inv_processing = self.datamodule.transform.inverse_preprocessing

        (
            images,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            visibilities,
            extra_coords,
        ) = batch

        stages_pred_heatmaps = self.model(images)

        hm_loss, tags_loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
        )
        loss = hm_loss + tags_loss
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        losses = {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "tags_loss": tags_loss.item(),
        }
        metrics_params = dict(
            step=self.current_step, epoch=self.current_epoch, split=self.stage
        )
        self.steps_metrics.append(losses, **metrics_params)

        if self.is_log_step and "eval" in self.stage:
            numpy_images = inv_processing(images.detach().cpu().numpy())

            results = self.ResultsClass.from_preds(
                numpy_images,
                stages_target_heatmaps,
                stages_pred_heatmaps,
                target_keypoints,
                extra_coords,
                det_thr=0.2,
            )
            self.results[self.stage] = results
            metrics = results.evaluate()
            self.validation_metrics.append(metrics | losses, **metrics_params)
