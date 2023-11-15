from torch import optim
from src.data import DataModule
from src.data.transforms.keypoints import KeypointsTransform
from src.data.datasets.keypoints import (
    SingleObjectKeypointsDataset,
    MultiObjectsKeypointsDataset,
)
from src.model.loss.keypoints import KeypointsLoss
from src.model.model.keypoints import KeypointsModel

from src.model.architectures.keypoints import HourglassNet
from src.model.architectures.keypoints.hourglass2 import HourglassNet as HN
from src.model.module.keypoints import KeypointsModule
from src.model.metrics.keypoints import KeypointsMetrics

from src.logging import get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    BaseCallback,
    KeypointsExamplesPlotterCallback,
    SaveLastAsOnnx,
)
from src.bin.config import Config
from src.utils.config import DS_ROOT
import geda.data_providers as gdp

log = get_pylogger(__name__)

ds2labels = {"COCO": gdp.coco.LABELS, "MPII": gdp.mpii.LABELS}


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


def create_datamodule(cfg: Config) -> DataModule:
    log.info("..Creating DataModule..")
    transform = KeypointsTransform(**cfg.dataloader.transform.to_dict())
    ds_root = str(DS_ROOT / cfg.setup.dataset / "HumanPose")
    labels = ds2labels[cfg.setup.dataset]
    if transform.multi_obj:
        DatasetClass = MultiObjectsKeypointsDataset
    else:
        DatasetClass = SingleObjectKeypointsDataset
    train_ds = DatasetClass(ds_root, "train", transform.train, labels)
    val_ds = DatasetClass(ds_root, "val", transform.train, labels)
    return DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=cfg.dataloader.batch_size,
    )


def create_model(cfg: Config) -> KeypointsModel:
    # net = HourglassNet(num_stages=2, num_keypoints=17)
    net = HN(num_stacks=2, num_classes=17)
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
