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
from ..transforms import KeypointsTransform
from ..datasets import SingleObjectKeypointsDataset, MultiObjectsKeypointsDataset
from ..loss import KeypointsLoss
from ..model import KeypointsModel
from ..architectures.hourglass import HourglassNet
from ..architectures.hourglass2 import HourglassNet as HN
from ..architectures.hrnet import PoseHigherResolutionNet
from ..module import KeypointsModule
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
    transform = KeypointsTransform(**cfg.dataloader.transform.to_dict())
    ds_root = str(DS_ROOT / ds_name / "HumanPose")
    labels = ds2labels[ds_name]
    limbs = ds2limbs[ds_name]

    if transform.multi_obj:
        DatasetClass = MultiObjectsKeypointsDataset
    else:
        DatasetClass = SingleObjectKeypointsDataset
    output_sizes = [(64, 64), (128, 128)]
    output_sizes = [(64, 64), (64, 64)]

    train_ds = DatasetClass(ds_root, "train", transform, output_sizes, labels, limbs)
    val_ds = DatasetClass(ds_root, "val", transform, output_sizes, labels, limbs)
    return KeypointsDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=cfg.dataloader.batch_size,
    )


def create_model(cfg: Config) -> KeypointsModel:
    num_kpts = 16 if cfg.setup.dataset == "MPII" else 17
    net = HourglassNet(num_stages=2, num_keypoints=num_kpts)
    # net = HN(num_stacks=2, num_classes=num_kpts)
    # net = PoseHigherResolutionNet(num_keypoints=num_kpts)
    model = KeypointsModel(net)

    return model


def create_module(cfg: Config) -> KeypointsModule:
    log.info("..Creating Module..")
    loss_fn = KeypointsLoss()
    model = create_model(cfg)
    optimizer = optim.Adam(model.parameters(), **cfg.optimizer.to_dict())
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[10, 20, 30], gamma=0.1
    # )
    labels = ds2labels[cfg.setup.dataset]
    metrics = KeypointsMetrics()
    module = KeypointsModule(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        labels=labels,
        optimizers={"optim": optimizer},
        schedulers={},
    )
    return module
