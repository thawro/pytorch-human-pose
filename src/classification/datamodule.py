"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule
from src.classification.transforms import ClassificationTransform
from src.classification.datasets import ClassificationDataset


class ClassificationDataModule(DataModule):
    train_ds: ClassificationDataset
    val_ds: ClassificationDataset
    test_ds: ClassificationDataset
    transform: ClassificationTransform


if __name__ == "__main__":
    from src.classification.datasets import ImageNetClassificationDataset
    from src.classification.transforms import ClassificationTransform

    transform = ClassificationTransform()
    train_ds = ImageNetClassificationDataset("data/ImageNet", "train", transform)
    val_ds = ImageNetClassificationDataset("data/ImageNet", "val", transform)

    datamodule = ClassificationDataModule(
        train_ds,
        val_ds,
        None,
        transform,
        batch_size=12,
        num_workers=16,
        pin_memory=False,
    )
    datamodule.explore()
