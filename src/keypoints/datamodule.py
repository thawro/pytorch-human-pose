"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule

from .datasets.coco import CocoKeypointsDataset


class KeypointsDataModule(DataModule):
    train_ds: CocoKeypointsDataset
    val_ds: CocoKeypointsDataset
    test_ds: CocoKeypointsDataset
