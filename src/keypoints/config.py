from dataclasses import dataclass

from torch import nn

from src.base.callbacks import BaseCallback
from src.base.config import BaseConfig, DataloaderConfig, DatasetConfig, TransformConfig
from src.base.datamodule import DataModule
from src.keypoints.architectures.original_higher_hrnet import get_pose_net
from src.logger.pylogger import log
from src.utils.config import DS_ROOT

from .architectures import AEHourglassNet, HigherHRNet
from .callbacks import KeypointsExamplesPlotterCallback
from .datasets.coco_keypoints import CocoKeypoints, collate_fn
from .datasets.transforms import KeypointsTransform
from .loss import AEKeypointsLoss
from .model import AEKeypointsModel
from .module import MPPEKeypointsModule


@dataclass
class KeypointsTransformConfig(TransformConfig):
    symmetric_keypoints: list[int]


@dataclass
class KeypointsDatasetConfig(DatasetConfig):
    pass


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
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def create_datamodule(self) -> DataModule:
        log.info("..Creating KeypointsDataModule..")
        transform = KeypointsTransform("COCO")
        train_ds = CocoKeypoints(
            "data/COCO/raw", "train2017", transform.random, (512, 512), [1 / 4, 1 / 2]
        )
        val_ds = CocoKeypoints(
            "data/COCO/raw", "val2017", transform.inference, (512, 512), [1 / 4, 1 / 2]
        )
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
        num_kpts = 17
        if arch == "Hourglass":
            net = AEHourglassNet(num_kpts, num_stages=2)
        elif arch == "HigherHRNet":
            net = HigherHRNet(num_kpts, C=32)
        elif arch == "OriginalHigherHRNet":
            log.warn("USING ORIGINAL HIGHER HRNET")
            imagenet_ckpt_path = (
                "/home/thawro/Desktop/projects/pytorch-human-pose/pretrained/hrnet_w32-36af842e.pth"
            )
            init_weights = (
                self.setup.ckpt_path is None and self.setup.pretrained_ckpt_path is not None
            )
            net = get_pose_net(init_weights, imagenet_ckpt_path, self.setup.is_train)
        else:
            raise ValueError("MPPE implemented only for Hourglass and HigherHRNet")
        return net

    def _create_model(self) -> AEKeypointsModel | nn.Module:
        log.info("..Creating Model..")
        net = self.create_net()

        if self.setup.is_train:
            return AEKeypointsModel(net, num_keypoints=17)
        else:
            return net

    def create_module(self) -> MPPEKeypointsModule:
        log.info("..Creating MPPEKeypointsModule..")
        loss_fn = AEKeypointsLoss([1 / 4, 1 / 2])
        model = self._create_model()
        module = MPPEKeypointsModule(
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
