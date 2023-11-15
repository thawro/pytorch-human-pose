"""Dataset classes"""

from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import glob


class BaseDataset(Dataset):
    root: Path

    def __init__(self, root: str, split: str, transform: A.Compose):
        self.transform = transform
        self.split = split
        self.root = Path(root)


class BaseImageDataset(BaseDataset):
    def __init__(self, root: str, split: str, transform: A.Compose):
        super().__init__(root, split, transform)
        self.images_filepaths = sorted(glob.glob(f"{str(self.root)}/images/{split}/*"))

    def load_image(self, idx: int) -> np.ndarray:
        image = np.asarray(Image.open(self.images_filepaths[idx]))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
