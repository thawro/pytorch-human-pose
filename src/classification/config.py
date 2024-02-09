from src.base.config import BaseConfig
from dataclasses import dataclass
from torch import nn

from .datamodule import ClassificationDataModule
from .datasets import ImageNetClassificationDataset
from .model import ClassificationModel
from .module import ClassificationModule
from .transforms import ClassificationTransform
from .architectures.hrnet import ClassificationHRNet
from .loss import ClassificationLoss

from src.utils.config import DS_ROOT


@dataclass
class ClassificationConfig(BaseConfig):

    def create_datamodule(self) -> ClassificationDataModule:
        ds_root = str(DS_ROOT / self.dataloader.dataset.name)
        transform = ClassificationTransform(
            **self.dataloader.dataset.transform.to_dict()
        )

        train_ds = ImageNetClassificationDataset(ds_root, "train", transform)
        val_ds = ImageNetClassificationDataset(ds_root, "val", transform)

        self.labels = train_ds.labels
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
        return ClassificationHRNet(C=32, num_classes=1000)

    def _create_model(self) -> ClassificationModel:
        net = self.create_net()
        return ClassificationModel(net)

    def create_module(self) -> ClassificationModule:
        model = self._create_model()
        loss_fn = ClassificationLoss()
        module = ClassificationModule(model=model, loss_fn=loss_fn, labels=self.labels)
        return module
