"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule
from src.classification.datasets.imagenet import ImagenetClassificationDataset


class ClassificationDataModule(DataModule):
    train_ds: ImagenetClassificationDataset
    val_ds: ImagenetClassificationDataset
    test_ds: ImagenetClassificationDataset


if __name__ == "__main__":
    from src.classification.transforms import ClassificationTransform

    transform = ClassificationTransform()
    train_ds = ImagenetClassificationDataset("data/ImageNet", "train", transform.train)
    val_ds = ImagenetClassificationDataset("data/ImageNet", "val", transform.inference)

    datamodule = ClassificationDataModule(
        train_ds,
        val_ds,
        None,
        batch_size=12,
        num_workers=16,
        pin_memory=False,
    )
    datamodule.explore()
