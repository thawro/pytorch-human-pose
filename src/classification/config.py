from dataclasses import dataclass

import torchvision.datasets as datasets
from torch import nn

from src.base.callbacks import BaseCallback
from src.base.config import BaseConfig
from src.logger.pylogger import log
from src.utils.config import DS_ROOT

from .architectures import ClassificationHRNet
from .datamodule import ClassificationDataModule
from .loss import ClassificationLoss
from .model import ClassificationModel
from .module import ClassificationModule
from .transforms import ClassificationTransform


@dataclass
class ClassificationConfig(BaseConfig):
    def create_datamodule(self) -> ClassificationDataModule:
        log.info("..Creating ClassificationDataModule..")

        transform = ClassificationTransform(**self.transform.to_dict())

        train_ds_root = str(
            DS_ROOT / self.dataloader.train_ds.name / self.dataloader.train_ds.split
        )
        val_ds_root = str(DS_ROOT / self.dataloader.val_ds.name / self.dataloader.val_ds.split)

        train_ds = datasets.ImageFolder(train_ds_root, transform.train)
        val_ds = datasets.ImageFolder(val_ds_root, transform.inference)
        self.labels = []

        return ClassificationDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
        )

    def create_net(self) -> nn.Module:
        log.info(f"..Creating {self.model.architecture}..")
        if self.model.architecture == "HRNet":
            return ClassificationHRNet(C=32, num_classes=1000)
        else:
            raise ValueError("Wrong architecture type")

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
