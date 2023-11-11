from torch import optim
from src.data import DataModule
from src.data.transforms.dummy import DummyTransform
from src.data.datasets.dummy import DummyDataset
from src.model.loss.dummy import DummyLoss
from src.model.model.dummy import DummyModel
from src.model.module.dummy import DummyModule
from src.logging import get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    BaseCallback,
    DummyExamplesPlotterCallback,
)
from src.model.module import BaseModule
from src.model.metrics import DummyMetrics
from src.bin.config import Config

log = get_pylogger(__name__)


def create_datamodule(cfg: Config) -> DataModule:
    log.info("..Creating DataModule..")
    transform = DummyTransform()
    train_ds = DummyDataset("", "train", transform=transform.train)
    val_ds = DummyDataset("", "val", transform=transform.train)
    return DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=cfg.dataloader.batch_size,
    )


def create_module(cfg: Config) -> BaseModule:
    log.info("..Creating Module..")
    loss_fn = DummyLoss()
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), **cfg.optimizer.to_dict())
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 30], gamma=0.1
    )
    metrics = DummyMetrics()
    module = DummyModule(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizers={"optim": optimizer},
        schedulers={"optim": scheduler},
    )
    return module


def create_callbacks(cfg: Config) -> list[BaseCallback]:
    log.info("..Creating Callbacks..")
    callbacks = [
        DummyExamplesPlotterCallback("dummy", ["train", "val"]),
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        ModelSummary(depth=4),
        SaveModelCheckpoint(name="best", metric="MAE", mode="min", stage="val"),
        SaveModelCheckpoint(name="last", last=True, top_k=0, stage="val"),
    ]
    if cfg.setup.ckpt_path is not None:
        callbacks.append(LoadModelCheckpoint(cfg.setup.ckpt_path))
    return callbacks
