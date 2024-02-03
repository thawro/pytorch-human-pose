"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule
from .transforms import ClassificationTransform
from .datasets import ClassificationDataset


class ClassificationDataModule(DataModule):
    train_ds: ClassificationDataset
    val_ds: ClassificationDataset
    test_ds: ClassificationDataset
    transform: ClassificationTransform
