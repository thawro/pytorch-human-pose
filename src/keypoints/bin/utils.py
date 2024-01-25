from torch import optim
import torch
from src.logging import get_pylogger
from src.base.callbacks import (
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    BaseCallback,
    MetricsLogger,
    SaveLastAsOnnx,
)
from src.base.lr_scheduler import LRScheduler

from ..datamodule import KeypointsDataModule
from ..datasets import BaseKeypointsDataset, collate_fn
from ..transforms import KeypointsTransform

from ..loss import KeypointsLoss, AEKeypointsLoss
from ..model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from ..architectures.hourglass import HourglassNet, AEHourglassNet
from ..architectures.simple_baseline import SimpleBaseline
from ..architectures.hrnet import HRNet
from ..architectures.higher_hrnet import HigherHRNet
from ..module import SPPEKeypointsModule, MPPEKeypointsModule
from ..callbacks import KeypointsExamplesPlotterCallback
from ..config import Config
from src.utils.fp16_utils.fp16util import network_to_half
from src.keypoints.architectures.original_higher_hrnet import get_pose_net


from torch.nn.parallel import DistributedDataParallel as DDP


log = get_pylogger(__name__)


def create_callbacks(cfg: Config) -> list[BaseCallback]:
    log.info("..Creating Callbacks..")
    callbacks = [
        KeypointsExamplesPlotterCallback("keypoints"),
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        MetricsLogger(),
        ModelSummary(depth=4),
        SaveModelCheckpoint(
            name="best", metric="loss", last=True, mode="min", stage="val"
        ),
    ]

    if not cfg.is_debug:
        callbacks.extend(
            [
                # SaveLastAsOnnx(every_n_minutes=60),
            ]
        )
    return callbacks


def create_datamodule(cfg: Config) -> KeypointsDataModule:
    ds_cfg = cfg.dataset
    log.info("..Creating DataModule..")

    transform: KeypointsTransform = ds_cfg.TransformClass(
        **cfg.dataloader.transform.to_dict()
    )

    train_ds: BaseKeypointsDataset = ds_cfg.DatasetClass(
        ds_cfg.root, "train", transform, cfg.hm_resolutions
    )
    val_ds: BaseKeypointsDataset = ds_cfg.DatasetClass(
        ds_cfg.root, "val", transform, cfg.hm_resolutions
    )
    return KeypointsDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=cfg.dataloader.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,
    )


def create_model(cfg: Config) -> BaseKeypointsModel:
    num_kpts = cfg.num_keypoints
    arch = cfg.setup.arch
    is_sppe = cfg.is_sppe

    if is_sppe:
        ModelClass = KeypointsModel
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
        ModelClass = AEKeypointsModel
        if arch == "Hourglass":
            net = AEHourglassNet(num_kpts, num_stages=2)
        elif arch == "HigherHRNet":
            net = HigherHRNet(num_kpts, C=32)
        elif arch == "OriginalHigherHRNet":
            log.warn("USING ORIGINAL HIGHER HRNET")
            imagenet_ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/pretrained/hrnet_w32-36af842e.pth"
            init_weights = cfg.setup.ckpt_path is None
            net = get_pose_net(init_weights, imagenet_ckpt_path, cfg.setup.is_train)
            # net = get_pose_net(False, imagenet_ckpt_path, False)
            # org_ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/pretrained/pose_higher_hrnet_w32_512.pth"
            # net.load_state_dict(torch.load(org_ckpt_path), strict=True)

        else:
            raise ValueError("MPPE implemented only for Hourglass and HigherHRNet")

    # for multi-GPU DDP
    if cfg.setup.is_train:
        # if fp16_enabled
        net = network_to_half(net)
    if cfg.setup.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net.cuda(cfg.trainer.device_id), device_ids=[cfg.trainer.device_id])
    if cfg.setup.is_train:
        return ModelClass(net, num_keypoints=cfg.num_keypoints)
    else:
        return net


def create_module(
    cfg: Config, model: BaseKeypointsModel, labels: list[str]
) -> SPPEKeypointsModule | MPPEKeypointsModule:
    log.info("..Creating Module..")
    if cfg.is_sppe:
        loss_fn = KeypointsLoss().cuda(cfg.trainer.device_id)
    else:
        loss_fn = AEKeypointsLoss(cfg.hm_resolutions).cuda(cfg.trainer.device_id)

    module = cfg.ModuleClass(
        model=model, loss_fn=loss_fn, labels=labels, limbs=cfg.dataset.limbs
    )
    return module
