from dataclasses import dataclass
from torch import nn
from typing import Literal, Type

from src.logger.pylogger import log

from .datamodule import KeypointsDataModule
from .datasets import (
    BaseKeypointsDataset,
    SppeCocoDataset,
    SppeMpiiDataset,
    MppeCocoDataset,
    MppeMpiiDataset,
    collate_fn,
)
from .module import BaseKeypointsModule, SPPEKeypointsModule, MPPEKeypointsModule
from .transforms import SPPEKeypointsTransform, MPPEKeypointsTransform
from .model import KeypointsModel, AEKeypointsModel, BaseKeypointsModel
from .architectures import (
    HourglassNet,
    AEHourglassNet,
    SimpleBaseline,
    HigherHRNet,
    HRNet,
)
from .loss import KeypointsLoss, AEKeypointsLoss
from .callbacks import KeypointsExamplesPlotterCallback


from src.base.config import BaseConfig, TransformConfig, DatasetConfig, DataloaderConfig
from src.keypoints.architectures.original_higher_hrnet import get_pose_net
from src.utils.config import DS_ROOT
from src.logger.pylogger import log
from src.base.callbacks import BaseCallback



@dataclass
class KeypointsTransformConfig(TransformConfig):
    symmetric_keypoints: list[int]


@dataclass
class KeypointsDatasetConfig(DatasetConfig):
    mode: Literal["SPPE", "MPPE"]

    @property
    def DatasetClass(self) -> Type[BaseKeypointsDataset]:
        Datasets = {
            "SPPE": {"COCO": SppeCocoDataset, "MPII": SppeMpiiDataset},
            "MPPE": {"COCO": MppeCocoDataset, "MPII": MppeMpiiDataset},
        }
        return Datasets[self.mode][self.name]

    @property
    def TransformClass(self) -> Type[SPPEKeypointsTransform | MPPEKeypointsTransform]:
        if self.mode == "SPPE":
            return SPPEKeypointsTransform
        elif self.mode == "MPPE":
            return MPPEKeypointsTransform
        else:
            raise ValueError("Wrong mode passed. Possible: ['SPPE', 'MPPE']")

    @property
    def subdir(self) -> str:
        if self.mode == "SPPE":
            return "SPPEHumanPose"
        elif self.mode == "MPPE":
            return "HumanPose"
        else:
            raise ValueError("Wrong mode passed. Possible: ['SPPE', 'MPPE']")

    @property
    def root(self) -> str:
        return str(DS_ROOT / self.name / self.subdir)


@dataclass
class KeypointsDataloaderConfig(DataloaderConfig):
    batch_size: int
    pin_memory: bool
    num_workers: int
    dataset: KeypointsDatasetConfig


@dataclass
class KeypointsConfig(BaseConfig):
    dataloader: KeypointsDataloaderConfig

    @property
    def is_sppe(self) -> bool:
        return self.dataloader.dataset.mode == "SPPE"

    @property
    def is_mpii(self) -> bool:
        return self.dataloader.dataset.name == "MPII"

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    @property
    def num_keypoints(self) -> int:
        if self.dataloader.dataset.name == "MPII":
            return 16
        elif self.dataloader.dataset.name == "COCO":
            return 17
        else:
            raise ValueError("Wrong dataset name passed. Possible: ['MPII', 'COCO']")

    @property
    def hm_resolutions(self) -> list[float]:
        if self.is_sppe:
            return [1 / 4, 1 / 4]
        else:
            if self.model.architecture in ["HigherHRNet", "OriginalHigherHRNet"]:
                return [1 / 4, 1 / 2]
            elif self.model.architecture == "Hourglass":
                return [1 / 4, 1 / 4]  # for Hourglass
            else:
                raise ValueError(
                    "For MPPE mode there are only HigherHRNet and Hourglass networks available"
                )

    @property
    def ModuleClass(self) -> Type[SPPEKeypointsModule | MPPEKeypointsModule]:
        if self.is_sppe:
            return SPPEKeypointsModule
        else:
            return MPPEKeypointsModule

    def create_datamodule(self) -> KeypointsDataModule:
        log.info("..Creating KeypointsDataModule..")
        ds_cfg = self.dataloader.dataset

        transform = ds_cfg.TransformClass(
            **ds_cfg.transform.to_dict(),
            symmetric_keypoints=ds_cfg.DatasetClass.symmetric_labels,
        )

        train_ds: BaseKeypointsDataset = ds_cfg.DatasetClass(
            ds_cfg.root, "train", transform, self.hm_resolutions
        )
        val_ds: BaseKeypointsDataset = ds_cfg.DatasetClass(
            ds_cfg.root, "val", transform, self.hm_resolutions
        )
        self.labels = train_ds.labels
        self.limbs = train_ds.limbs
        return KeypointsDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            transform=transform,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            collate_fn=collate_fn,
        )

    def create_net(self) -> nn.Module:
        log.info("..Creating Net..")
        arch = self.model.architecture
        is_sppe = self.is_sppe
        num_kpts = self.num_keypoints

        if is_sppe:
            if arch == "Hourglass":
                net = HourglassNet(num_kpts, num_stages=8)
            elif arch == "SimpleBaseline":
                net = SimpleBaseline(num_keypoints=num_kpts, backbone="resnet101")
            elif arch == "HRNet":
                net = HRNet(num_keypoints=num_kpts, C=32)
            else:
                raise ValueError(
                    "SPPE implemented only for Hourglass, SimpleBaseline and HRNet"
                )
        else:
            if arch == "Hourglass":
                net = AEHourglassNet(num_kpts, num_stages=2)
            elif arch == "HigherHRNet":
                net = HigherHRNet(num_kpts, C=32)
            elif arch == "OriginalHigherHRNet":
                log.warn("USING ORIGINAL HIGHER HRNET")
                imagenet_ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/pretrained/hrnet_w32-36af842e.pth"
                init_weights = self.setup.ckpt_path is None
                net = get_pose_net(
                    init_weights, imagenet_ckpt_path, self.setup.is_train
                )
            else:
                raise ValueError("MPPE implemented only for Hourglass and HigherHRNet")
        return net

    def _create_model(self) -> BaseKeypointsModel | nn.Module:
        ModelClass = KeypointsModel if self.is_sppe else AEKeypointsModel
        log.info(f"..Creating {ModelClass.__name__}..")
        log.info("..Creating Model..")
        net = self.create_net()

        if self.setup.is_train:
            return ModelClass(net, num_keypoints=self.num_keypoints)
        else:
            return net

    def create_module(self) -> BaseKeypointsModule:
        log.info(f"..Creating {self.ModuleClass.__name__}..")
        if self.is_sppe:
            loss_fn = KeypointsLoss()
        else:
            loss_fn = AEKeypointsLoss(self.hm_resolutions)
        model = self._create_model()
        module = self.ModuleClass(
            model=model,
            loss_fn=loss_fn,
            labels=self.labels,
            limbs=self.limbs,
        )
        return module

    def create_callbacks(self) -> list[BaseCallback]:
        callbacks = super().create_callbacks()
        callbacks.append(KeypointsExamplesPlotterCallback("keypoints"))
        return callbacks
