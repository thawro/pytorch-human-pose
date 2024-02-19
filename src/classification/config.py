from src.base.config import BaseConfig
from dataclasses import dataclass
from torch import nn
from src.logger.pylogger import log
from .datamodule import ClassificationDataModule
from .model import ClassificationModel
from .module import ClassificationModule
from .transforms import ClassificationTransform
from .architectures.hrnet import ClassificationHRNet
from .architectures.original_hrnet import OriginalHRNet
from .loss import ClassificationLoss

from src.utils.config import DS_ROOT
from src.base.callbacks import BaseCallback

import torchvision.datasets as datasets


@dataclass
class ClassificationConfig(BaseConfig):

    def create_datamodule(self) -> ClassificationDataModule:
        log.info("..Creating ClassificationDataModule..")
        ds_root = str(DS_ROOT / self.dataloader.dataset.name)
        transform = ClassificationTransform(
            **self.dataloader.dataset.transform.to_dict()
        )

        train_ds = datasets.ImageFolder(
            str(DS_ROOT / ds_root / "train"), transform.random
        )
        val_ds = datasets.ImageFolder(
            str(DS_ROOT / ds_root / "val"), transform.inference
        )
        self.labels = []

        return ClassificationDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            transform=transform,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
        )

    def create_net(self) -> nn.Module:
        log.info(f"..Creating {self.model.architecture}..")
        if self.model.architecture == "HRNet":
            return ClassificationHRNet(C=32, num_classes=1000)
        elif self.model.architecture == "OriginalHRNet":
            return OriginalHRNet(C=32)
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
