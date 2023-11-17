"""Implementation of specialized Module"""
import torch
from torch import optim
from torch import Tensor

from src.base.module import BaseModule

from .model import KeypointsModel
from .loss import KeypointsLoss
from .results import KeypointsResult
from .metrics import KeypointsMetrics


class KeypointsModule(BaseModule):
    model: KeypointsModel
    loss_fn: KeypointsLoss
    metrics: KeypointsMetrics
    results: dict[str, KeypointsResult]

    def __init__(
        self,
        model: KeypointsModel,
        loss_fn: KeypointsLoss,
        metrics: KeypointsMetrics,
        labels: list[str],
        optimizers: dict[str, optim.Optimizer],
        schedulers: dict[str, optim.lr_scheduler.LRScheduler] = {},
    ):
        super().__init__(model, loss_fn, metrics, optimizers, schedulers)
        self.labels = labels

    def _common_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, update_metrics: bool
    ):
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()

        images, target_heatmaps, target_weights = batch

        stages_pred_heatmaps = self.model(images)

        loss = self.loss_fn.calculate_loss(
            stages_pred_heatmaps, target_heatmaps, target_weights
        )
        if self.stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        if update_metrics:
            losses = {"loss": loss.item()}
            # metrics = self.metrics.calculate_metrics(
            #     stages_pred_heatmaps.mean(dim=1), target_heatmaps
            # )
            # self.steps_metrics_storage.append(metrics, self.stage)
            # self.current_epoch_steps_metrics_storage.append(metrics, self.stage)
            self.steps_metrics_storage.append(losses, self.stage)
            self.current_epoch_steps_metrics_storage.append(losses, self.stage)

        log_step = self.current_step % self.log_every_n_steps == 0
        log_epoch = (batch_idx + 1) == self.total_batches[self.stage]
        if log_step or log_epoch:
            inv_processing = self.datamodule.transform.inverse_preprocessing
            numpy_images = inv_processing(images.detach().cpu().numpy())
            h, w = numpy_images.shape[:2]
            # TODO: assuming that all hms are same size
            import torchvision.transforms.functional as F

            num_stages = len(stages_pred_heatmaps)
            pred_heatmaps = torch.stack(
                [
                    F.resize(stages_pred_heatmaps[i], [h, w], antialias=True)
                    for i in range(num_stages)
                ]
            ).mean(dim=0)
            self.results[self.stage] = KeypointsResult(
                images=numpy_images,
                target_heatmaps=target_heatmaps[-1].cpu().numpy(),
                pred_heatmaps=pred_heatmaps.detach().cpu().numpy(),
                keypoints=None,
            )
