"""Implementation of specialized Module"""

from torch import optim, Tensor

from src.base.module import BaseModule

from .model import ClassificationModel
from .loss import ClassificationLoss
from .datamodule import ClassificationDataModule
from .results import ClassificationResult
from src.base.lr_scheduler import LRScheduler
from src.utils.fp16_utils.fp16_optimizer import FP16_Optimizer


_batch = tuple[Tensor, list[dict], Tensor]


class BaseClassificationModule(BaseModule):
    datamodule: ClassificationDataModule
    use_fp16: bool

    def __init__(
        self,
        model: ClassificationModel,
        loss_fn: ClassificationLoss,
        labels: list[str],
        use_fp16: bool,
    ):
        super().__init__(model, loss_fn, use_fp16)
        self.labels = labels

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        optimizer = optim.Adam(self.model.parameters(), lr=1e-1)
        if self.use_fp16:
            optimizer = FP16_Optimizer(
                optimizer, dynamic_loss_scale=True, verbose=False
            )
            _optimizer = optimizer.optimizer
        else:
            _optimizer = optimizer

        optimizers = {"optim": optimizer}
        schedulers = {
            "optim": LRScheduler(
                optim.lr_scheduler.MultiStepLR(
                    _optimizer,
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

    def _common_step(
        self, batch: _batch, batch_idx: int
    ) -> dict[str, float] | tuple[dict[str, float], list[ClassificationResult]]:
        images, annots, targets = batch
        logits = self.model(images)
        loss = self.loss_fn.calculate_loss(targets, logits)
        if self.stage == "train":
            self.optimizers["optim"].zero_grad()
            if self.use_fp16:
                self.optimizers["optim"].backward(loss)
            else:
                loss.backward()
            self.optimizers["optim"].step()

        metrics = {"loss": loss.item()}

        if self.stage == "train":
            return metrics

        targets = targets.cpu().numpy()
        logits = logits.detach().cpu().numpy()
        results = []
        for i in range(len(images)):
            _image = self.datamodule.transform.inverse_preprocessing(images[i].detach())

            result = ClassificationResult(
                image=_image,
                target=targets[i],
                pred=logits[i],
            )
            results.append(result)
        return metrics, results
