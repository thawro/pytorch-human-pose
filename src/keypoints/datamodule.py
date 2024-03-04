"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule

from .datasets.coco_keypoints import CocoKeypoints


class KeypointsDataModule(DataModule):
    train_ds: CocoKeypoints
    val_ds: CocoKeypoints
    test_ds: CocoKeypoints
