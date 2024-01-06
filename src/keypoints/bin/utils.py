from torch import optim
import torch
from src.logging import get_pylogger
from src.utils.config import DS_ROOT
from src.base.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    BaseCallback,
    SaveLastAsOnnx,
)

from ..datamodule import KeypointsDataModule
from ..transforms import SPPEKeypointsTransform, MPPEKeypointsTransform
from ..datasets import (
    SppeMpiiDataset,
    MppeMpiiDataset,
    SppeCocoDataset,
    MppeCocoDataset,
    collate_fn,
)
from ..loss import KeypointsLoss, AEKeypointsLoss
from ..model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from ..architectures.hourglass import HourglassNet, AEHourglassNet
from ..architectures.simple_baseline import SimpleBaseline
from ..architectures.hrnet import HRNet
from ..architectures.higher_hrnet import HigherHRNet
from ..module import BaseKeypointsModule, SPPEKeypointsModule, MPPEKeypointsModule
from ..callbacks import KeypointsExamplesPlotterCallback
from ..config import Config
from ..results import (
    SppeMpiiKeypointsResults,
    SppeCocoKeypointsResults,
    MppeMpiiKeypointsResults,
    MppeCocoKeypointsResults,
)


log = get_pylogger(__name__)


def create_callbacks(cfg: Config) -> list[BaseCallback]:
    log.info("..Creating Callbacks..")
    callbacks = [
        KeypointsExamplesPlotterCallback("keypoints"),
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        ModelSummary(depth=4),
        SaveModelCheckpoint(
            name="best", metric="loss", last=True, mode="min", stage="eval_val"
        ),
    ]
    if cfg.setup.ckpt_path is not None:
        callbacks.append(LoadModelCheckpoint(cfg.setup.ckpt_path, lr=cfg.optimizer.lr))

    if not cfg.setup.is_debug:
        callbacks.extend(
            [
                # SaveLastAsOnnx(every_n_minutes=60),
            ]
        )
    return callbacks


def create_datamodule(cfg: Config) -> KeypointsDataModule:
    ds_name = cfg.setup.dataset
    log.info("..Creating DataModule..")
    Datasets = {
        "SPPE": {"COCO": SppeCocoDataset, "MPII": SppeMpiiDataset},
        "MPPE": {"COCO": MppeCocoDataset, "MPII": MppeMpiiDataset},
    }
    mode = "SPPE" if cfg.setup.is_sppe else "MPPE"
    DatasetClass = Datasets[mode][ds_name]

    if cfg.setup.is_sppe:
        TransformClass = SPPEKeypointsTransform
        ds_subdir = "SPPEHumanPose"
        hm_resolutions = [1 / 4, 1 / 4]
    else:
        TransformClass = MPPEKeypointsTransform
        ds_subdir = "HumanPose"
        hm_resolutions = [1 / 4, 1 / 2]

    ds_root = str(DS_ROOT / ds_name / ds_subdir)

    transform = TransformClass(**cfg.dataloader.transform.to_dict())

    train_ds = DatasetClass(ds_root, "train", transform, hm_resolutions)
    val_ds = DatasetClass(ds_root, "val", transform, hm_resolutions)
    return KeypointsDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=cfg.dataloader.batch_size,
        collate_fn=collate_fn,
    )


def create_model(cfg: Config) -> BaseKeypointsModel:
    num_kpts = cfg.setup.num_keypoints
    arch = cfg.setup.arch
    is_sppe = cfg.setup.is_sppe

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
        else:
            raise ValueError("MPPE implemented only for Hourglass and HigherHRNet")

    net = torch.nn.DataParallel(net, device_ids=[0, 1])

    model = ModelClass(net, num_keypoints=cfg.setup.num_keypoints)
    return model


def create_module(cfg: Config, labels: list[str]) -> BaseKeypointsModule:
    log.info("..Creating Module..")
    if cfg.setup.is_sppe:
        loss_fn = KeypointsLoss()
        ModuleClass = SPPEKeypointsModule
        if cfg.setup.is_mpii:
            ResultsClass = SppeMpiiKeypointsResults
        else:
            ResultsClass = SppeCocoKeypointsResults
    else:
        if cfg.setup.is_sppe:
            hm_resolutions = [1 / 4, 1 / 4]
        else:
            hm_resolutions = [1 / 4, 1 / 2]

        loss_fn = AEKeypointsLoss(hm_resolutions)
        ModuleClass = MPPEKeypointsModule
        if cfg.setup.is_mpii:
            ResultsClass = MppeMpiiKeypointsResults
        else:
            ResultsClass = MppeCocoKeypointsResults

    model = create_model(cfg)

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer.to_dict())
    scheduler = {}
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[10, 20, 30], gamma=0.1
    # )
    module = ModuleClass(
        model=model,
        loss_fn=loss_fn,
        labels=labels,
        optimizers={"optim": optimizer},
        schedulers=scheduler,  # {"optim": scheduler},
        ResultsClass=ResultsClass,
    )
    return module
