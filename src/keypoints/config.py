from dataclasses import dataclass
from typing import Literal

from torch import nn

from src.base.callbacks import BaseCallback
from src.base.config import BaseConfig, DataloaderConfig, DatasetConfig, TransformConfig
from src.base.datamodule import DataModule
from src.logger.pylogger import log

from .architectures import AEHourglassNet, HigherHRNet
from .callbacks import KeypointsExamplesPlotterCallback
from .datasets.coco_keypoints import CocoKeypoints, collate_fn
from .datasets.transforms import KeypointsTransform
from .loss import AEKeypointsLoss
from .model import KeypointsModel
from .module import MPPEKeypointsModule


@dataclass
class KeypointsTransformConfig(TransformConfig):
    hm_resolutions: list[float]
    max_rotation: int
    min_scale: float
    max_scale: float
    scale_type: Literal["short", "long"]
    max_translate: int


@dataclass
class KeypointsDatasetConfig(DatasetConfig):
    out_size: int
    hm_resolutions: list[float]


@dataclass
class KeypointsDataloaderConfig(DataloaderConfig):
    train_ds: KeypointsDatasetConfig
    val_ds: KeypointsDatasetConfig


@dataclass
class KeypointsConfig(BaseConfig):
    dataloader: KeypointsDataloaderConfig
    transform: KeypointsTransformConfig
    num_kpts: int = 17

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def create_datamodule(self) -> DataModule:
        log.info("..Creating KeypointsDataModule..")
        transform = KeypointsTransform(**self.transform.to_dict())
        train_ds = CocoKeypoints(**self.dataloader.train_ds.to_dict(), transform=transform.train)
        val_ds = CocoKeypoints(**self.dataloader.val_ds.to_dict(), transform=transform.inference)
        self.labels = train_ds.labels
        self.limbs = train_ds.limbs
        return DataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            collate_fn=collate_fn,
        )

    def create_net(self) -> nn.Module:
        log.info(f"..Creating {self.model.architecture} Neural Network..")
        arch = self.model.architecture
        if arch == "Hourglass":
            net = AEHourglassNet(self.num_kpts, num_stages=2)
        elif arch == "HigherHRNet":
            net = HigherHRNet(self.num_kpts, C=32)
        else:
            raise ValueError("MPPE implemented only for Hourglass and HigherHRNet")
        return net

    def _create_model(self) -> KeypointsModel | nn.Module:
        log.info("..Creating Model..")
        net = self.create_net()

        if self.setup.is_train:
            return KeypointsModel(net, num_kpts=17)
        else:
            return net

    def create_module(self) -> MPPEKeypointsModule:
        log.info("..Creating MPPEKeypointsModule..")
        loss_fn = AEKeypointsLoss()
        model = self._create_model()
        module = MPPEKeypointsModule(
            model=model,
            loss_fn=loss_fn,
            labels=self.labels,
            limbs=self.limbs,
            optimizers=self.get_optimizers_params(),
            lr_schedulers=self.get_lr_schedulers_params(),
        )
        return module

    def create_callbacks(self) -> list[BaseCallback]:
        callbacks = super().create_callbacks()
        callbacks.append(KeypointsExamplesPlotterCallback("keypoints"))
        return callbacks
