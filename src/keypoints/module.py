"""Implementation of specialized Module"""
from torch import optim, Tensor
from typing import Type

from src.base.module import BaseModule

from .model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from .loss import KeypointsLoss, AEKeypointsLoss
from .results import SPPEKeypointsResults, MPPEKeypointsResults
from .datamodule import KeypointsDataModule
from src.base.lr_scheduler import LRScheduler


_batch = tuple[Tensor, list[Tensor], Tensor, list, list, list]


class BaseKeypointsModule(BaseModule):
    datamodule: KeypointsDataModule

    def __init__(
        self,
        model: BaseKeypointsModel,
        loss_fn: KeypointsLoss,
        labels: list[str],
        ResultsClass: Type[SPPEKeypointsResults | MPPEKeypointsResults],
    ):
        super().__init__(model, loss_fn)
        self.labels = labels
        self.ResultsClass = ResultsClass

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        optimizers = {"optim": optim.Adam(self.model.parameters(), lr=1e-3)}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    optimizers["optim"],
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
    results: dict[str, SPPEKeypointsResults]
    ResultsClass: Type[SPPEKeypointsResults]

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
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

        metrics = {"loss": loss.item()}

        if self.is_log_step and "eval" in self.stage:
            inv_processing = self.datamodule.transform.inverse_preprocessing
            numpy_images = inv_processing(images.detach().cpu().numpy())
            results = self.ResultsClass.from_preds(
                numpy_images,
                target_heatmaps,
                pred_heatmaps.detach(),
                target_keypoints,
                target_weights,
                extra_coords,
                det_thr=0.2,
            )
            self.results[self.stage] = results
            metrics.update(results.evaluate())
        return metrics


class MPPEKeypointsModule(BaseKeypointsModule):
    model: AEKeypointsModel
    loss_fn: AEKeypointsLoss
    results: dict[str, MPPEKeypointsResults]
    ResultsClass: Type[MPPEKeypointsResults]

    def _common_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        inv_processing = self.datamodule.transform.inverse_preprocessing

        (
            images,
            stages_target_heatmaps,
            target_weights,
            target_keypoints,
            target_visibilities,
            extra_coords,
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
            loss.backward()
            self.optimizers["optim"].step()

        metrics = {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "tags_loss": tags_loss.item(),
        }

        if self.is_log_step and "eval" in self.stage:
            numpy_images = inv_processing(images.detach().cpu().numpy())

            results = self.ResultsClass.from_preds(
                numpy_images,
                stages_target_heatmaps,
                stages_pred_kpts_heatmaps,
                stages_pred_tags_heatmaps,
                target_keypoints,
                target_weights,
                extra_coords,
                max_num_people=10,
                det_thr=0.2,
                tag_thr=1,
            )
            self.results[self.stage] = results
            metrics.update(results.evaluate())
        return metrics
