"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule
from .transforms import KeypointsTransform
from .datasets import BaseKeypointsDataset


class KeypointsDataModule(DataModule):
    train_ds: BaseKeypointsDataset
    val_ds: BaseKeypointsDataset
    test_ds: BaseKeypointsDataset
    transform: KeypointsTransform
