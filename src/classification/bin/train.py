from src.base.bin.train import train
from src.logging import TerminalLogger
from src.base.trainer import Trainer
from src.utils.model import seed_everything
from src.base.callbacks import (
    BaseCallback,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    MetricsLogger,
    ModelSummary,
    SaveModelCheckpoint,
)

import os
from src.utils.config import RESULTS_PATH, DS_ROOT, NOW
from src.utils.fp16_utils.fp16util import network_to_half

from src.classification.datamodule import ClassificationDataModule
from src.classification.transforms import ClassificationTransform
from src.classification.datasets import ImageNetClassificationDataset
from src.classification.model import ClassificationModel
from src.classification.architectures.hrnet import ClassificationHRNet
from src.classification.module import ClassificationModule
from src.classification.loss import ClassificationLoss

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

USE_DISTRIBUTED = True
USE_FP16 = False


def create_datamodule() -> ClassificationDataModule:
    ds_root = str(DS_ROOT / "ImageNet")
    transform = ClassificationTransform()

    train_ds = ImageNetClassificationDataset(ds_root, "train", transform)
    val_ds = ImageNetClassificationDataset(ds_root, "val", transform)
    return ClassificationDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        transform=transform,
        batch_size=96,
        pin_memory=True,
        num_workers=4,
        use_distributed=USE_DISTRIBUTED,
    )


def create_model(device_id: int = 0) -> ClassificationModel:
    net = ClassificationHRNet(17, 32, 1000)
    is_train = True

    if is_train and USE_FP16:
        net = network_to_half(net)

    if USE_DISTRIBUTED:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(
            net.cuda(device_id), device_ids=[device_id], find_unused_parameters=True
        )
    else:
        net.cuda(device_id)
    model = ClassificationModel(net)
    return model


def create_module(
    model: ClassificationModel, labels: list[str]
) -> ClassificationModule:
    loss_fn = ClassificationLoss()
    module = ClassificationModule(
        model=model, loss_fn=loss_fn, labels=labels, use_fp16=USE_FP16
    )
    return module


def create_callbacks() -> list[BaseCallback]:
    callbacks = [
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        MetricsLogger(),
        ModelSummary(depth=4),
        SaveModelCheckpoint(
            name="best", metric="loss", last=True, mode="min", stage="val"
        ),
    ]
    return callbacks


def train_fn():
    if "LOCAL_RANK" in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = 0
    seed = 42

    seed_everything(seed)

    datamodule = create_datamodule()

    model = create_model(device_id)

    labels = datamodule.train_ds.labels

    module = create_module(model, labels)

    logs_path = str(RESULTS_PATH / "debug" / "test" / NOW)
    logger = TerminalLogger(logs_path, config={})

    callbacks = create_callbacks()

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        device_id=device_id,
        max_epochs=100,
        limit_batches=-1,
        log_every_n_steps=-1,
        use_distributed=USE_DISTRIBUTED,
        use_fp16=USE_FP16,
    )
    trainer.fit(module, datamodule, ckpt_path=None)


def main():
    torch.autograd.set_detect_anomaly(True)
    train(
        train_fn=train_fn,
        use_distributed=USE_DISTRIBUTED,
        use_fp16=USE_FP16,
    )


if __name__ == "__main__":
    main()
