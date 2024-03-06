from dataclasses import dataclass
from typing import Literal, Type

from torch import nn

from src.base.callbacks import BaseCallback
from src.base.config import BaseConfig, DataloaderConfig, DatasetConfig, TransformConfig
from src.base.trainer import Trainer
from src.logger.pylogger import log

from .architectures import AEHourglassNet, HigherHRNet
from .callbacks import KeypointsExamplesPlotterCallback
from .datamodule import KeypointsDataModule
from .datasets.coco import CocoKeypointsDataset, collate_fn
from .loss import AEKeypointsLoss
from .model import KeypointsModel
from .module import KeypointsModule
from .trainer import KeypointsTrainer
from .transforms import KeypointsTransform


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
    num_kpts: int
    max_num_people: int
    sigma: float
    mosaic_probability: float


@dataclass
class KeypointsDataloaderConfig(DataloaderConfig):
    train_ds: KeypointsDatasetConfig
    val_ds: KeypointsDatasetConfig


@dataclass
class KeypointsConfig(BaseConfig):
    dataloader: KeypointsDataloaderConfig
    transform: KeypointsTransformConfig

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def create_datamodule(self) -> KeypointsDataModule:
        log.info("..Creating KeypointsDataModule..")
        transform = KeypointsTransform(**self.transform.to_dict())
        train_ds = CocoKeypointsDataset(
            **self.dataloader.train_ds.to_dict(), transform=transform.train
        )
        val_ds = CocoKeypointsDataset(
            **self.dataloader.val_ds.to_dict(), transform=transform.inference
        )
        self.labels = train_ds.labels
        self.limbs = train_ds.limbs
        return KeypointsDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            collate_fn=collate_fn,
            use_DDP=self.trainer.use_DDP,
        )

    @property
    def architectures(self) -> dict[str, Type[nn.Module]]:
        return {"Hourglass": AEHourglassNet, "HigherHRNet": HigherHRNet}

    def _create_model(self) -> KeypointsModel | nn.Module:
        log.info("..Creating Model..")
        net = self.create_net()

        if self.setup.is_train:
            return KeypointsModel(net)
        else:
            return net

    def create_module(self) -> KeypointsModule:
        log.info("..Creating MPPEKeypointsModule..")
        loss_fn = AEKeypointsLoss()
        model = self._create_model()
        module = KeypointsModule(
            model=model,
            loss_fn=loss_fn,
            labels=self.labels,
            limbs=self.limbs,
            optimizers=self.get_optimizers_params(),
            lr_schedulers=self.get_lr_schedulers_params(),
        )
        return module

    @property
    def TrainerClass(self) -> Type[Trainer]:
        return KeypointsTrainer

    def create_callbacks(self) -> list[BaseCallback]:
        base_callbacks = super().create_callbacks()
        kpts_callbacks = [
            KeypointsExamplesPlotterCallback("keypoints"),
        ]
        return base_callbacks + kpts_callbacks
