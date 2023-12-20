"""DataModule used to load DataLoaders"""

from src.logging import get_pylogger
from src.base.datamodule import DataModule
from .transforms import KeypointsTransform
from .datasets import BaseKeypointsDataset

log = get_pylogger(__name__)


class KeypointsDataModule(DataModule):
    train_ds: BaseKeypointsDataset
    val_ds: BaseKeypointsDataset
    test_ds: BaseKeypointsDataset
    transform: KeypointsTransform

    @property
    def get_metrics(self):
        return self.train_ds.get_metrics
