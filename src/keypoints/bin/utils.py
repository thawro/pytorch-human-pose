from torch import optim
import geda.data_providers as gdp

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
from ..datasets import SPPEKeypointsDataset, MPPEKeypointsDataset, mppe_collate_fn
from ..loss import KeypointsLoss, AEKeypointsLoss
from ..model import BaseKeypointsModel, KeypointsModel, AEKeypointsModel
from ..architectures.hourglass import HourglassNet, AEHourglassNet
from ..architectures.simple_baseline import SimpleBaseline
from ..architectures.hrnet import HRNet
from ..architectures.higher_hrnet import HigherHRNet
from ..module import BaseKeypointsModule, KeypointsModule, AEKeypointsModule
from ..metrics import KeypointsMetrics
from ..callbacks import KeypointsExamplesPlotterCallback
from ..config import Config


log = get_pylogger(__name__)

ds2labels = {"COCO": gdp.coco.LABELS, "MPII": gdp.mpii.LABELS}
ds2limbs = {"COCO": gdp.coco.LIMBS, "MPII": gdp.mpii.LIMBS}


def create_callbacks(cfg: Config) -> list[BaseCallback]:
    log.info("..Creating Callbacks..")
    callbacks = [
        KeypointsExamplesPlotterCallback("keypoints", ["train", "val"]),
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        ModelSummary(depth=4),
        # SaveModelCheckpoint(name="best", metric="MAE", mode="min", stage="val"),
        SaveModelCheckpoint(name="last", last=True, top_k=0, stage="val"),
        # SaveLastAsOnnx(every_n_minutes=60),
    ]
    if cfg.setup.ckpt_path is not None:
        callbacks.append(LoadModelCheckpoint(cfg.setup.ckpt_path))
    return callbacks


def create_datamodule(cfg: Config) -> KeypointsDataModule:
    ds_name = cfg.setup.dataset
    log.info("..Creating DataModule..")

    if cfg.setup.is_sppe:
        DatasetClass = SPPEKeypointsDataset
        TransformClass = SPPEKeypointsTransform
        collate_fn = None
        ds_subdir = "SPPEHumanPose"
        hm_resolutions = [1 / 4, 1 / 4]
    else:
        DatasetClass = MPPEKeypointsDataset
        TransformClass = MPPEKeypointsTransform
        collate_fn = mppe_collate_fn
        ds_subdir = "HumanPose"
        hm_resolutions = [1 / 2]

    ds_root = str(DS_ROOT / ds_name / ds_subdir)
    labels = ds2labels[ds_name]
    limbs = ds2limbs[ds_name]

    transform = TransformClass(**cfg.dataloader.transform.to_dict())

    train_ds = DatasetClass(ds_root, "train", transform, hm_resolutions, labels, limbs)
    val_ds = DatasetClass(ds_root, "val", transform, hm_resolutions, labels, limbs)
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
            net = HourglassNet(num_kpts, num_stages=2)
        elif arch == "SimpleBaseline":
            net = SimpleBaseline(num_keypoints=num_kpts, backbone="resnet34")
        elif arch == "HRNet":
            net = HRNet(num_keypoints=num_kpts)
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

    model = ModelClass(net)
    return model


def create_module(cfg: Config) -> BaseKeypointsModule:
    log.info("..Creating Module..")
    if cfg.setup.is_sppe:
        loss_fn = KeypointsLoss()
        ModuleClass = KeypointsModule
    else:
        loss_fn = AEKeypointsLoss()
        ModuleClass = AEKeypointsModule
    model = create_model(cfg)
    optimizer = optim.Adam(model.parameters(), **cfg.optimizer.to_dict())
    scheduler = None
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[10, 20, 30], gamma=0.1
    # )
    metrics = KeypointsMetrics()
    module = ModuleClass(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        labels=ds2labels[cfg.setup.dataset],
        optimizers={"optim": optimizer},
        schedulers={"optim": scheduler},
    )
    return module
