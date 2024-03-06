from dataclasses import dataclass
from typing import Type

from torch import nn

from src.base.callbacks import BaseCallback
from src.base.config import BaseConfig
from src.logger.pylogger import log

from .architectures import ClassificationHRNet
from .datamodule import ClassificationDataModule
from .datasets import ImagenetClassificationDataset
from .loss import ClassificationLoss
from .model import ClassificationModel
from .module import ClassificationModule
from .transforms import ClassificationTransform


@dataclass
class ClassificationConfig(BaseConfig):
    def create_datamodule(self) -> ClassificationDataModule:
        log.info("..Creating ClassificationDataModule..")

        transform = ClassificationTransform(**self.transform.to_dict())

        train_ds = ImagenetClassificationDataset(
            **self.dataloader.train_ds.to_dict(), transform=transform.train
        )
        val_ds = ImagenetClassificationDataset(
            **self.dataloader.val_ds.to_dict(), transform=transform.inference
        )
        self.labels = []

        return ClassificationDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            use_DDP=self.trainer.use_DDP,
        )

    @property
    def architectures(self) -> dict[str, Type[nn.Module]]:
        return {"ClassificationHRNet": ClassificationHRNet}

    def _create_model(self) -> ClassificationModel:
        log.info("..Creating ClassificationModel..")
        net = self.create_net()
        return ClassificationModel(net)

    def create_module(self) -> ClassificationModule:
        log.info("..Creating ClassificationModule..")
        model = self._create_model()
        loss_fn = ClassificationLoss()
        module = ClassificationModule(model=model, loss_fn=loss_fn, labels=self.labels)
        return module

    def create_callbacks(self) -> list[BaseCallback]:
        return super().create_callbacks()
