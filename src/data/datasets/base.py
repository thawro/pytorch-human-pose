"""Dataset classes"""

import torchvision.datasets
import torchvision.transforms as T
import albumentations as A
from pathlib import Path

_transform = A.Compose | T.Compose | None


class BaseDataset(torchvision.datasets.VisionDataset):
    root: Path

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: _transform = None,
        target_transform: _transform = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.root = Path(root)
