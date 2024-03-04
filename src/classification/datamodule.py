"""DataModule used to load DataLoaders"""

from torchvision.datasets import ImageFolder

from src.base.datamodule import DataModule


class ClassificationDataModule(DataModule):
    train_ds: ImageFolder
    val_ds: ImageFolder
    test_ds: ImageFolder


if __name__ == "__main__":
    from src.classification.transforms import ClassificationTransform

    transform = ClassificationTransform()
    train_ds = ImageFolder("data/ImageNet/train", transform.train)
    val_ds = ImageFolder("data/ImageNet/val", transform.inference)

    datamodule = ClassificationDataModule(
        train_ds,
        val_ds,
        None,
        batch_size=12,
        num_workers=16,
        pin_memory=False,
    )
    datamodule.explore()
