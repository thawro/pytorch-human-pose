"""Dataset classes"""

from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import glob
from typing import Callable, Any


class BaseDataset(Dataset):
    root: Path

    def __init__(self, root: str, split: str, transform: A.Compose):
        self.transform = transform
        self.split = split
        self.root = Path(root)

    def plot(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def explore(self, idx: int = 0, callback: Callable[[int], Any] | None = None):
        if callback is not None:
            callback(idx)
        image = self.plot(idx)
        cv2.imshow("Sample", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        k = cv2.waitKeyEx(0)
        # change according to your system
        left_key = 65361
        right_key = 65363
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing")
            cv2.destroyAllWindows()
            return
        elif k % 256 == 32 or k == right_key:  # SPACE or right arrow pressed
            print("Space or right arrow hit, exploring next sample")
            idx += 1
        elif k == left_key:  # SPACE or right arrow pressed
            print("Left arrow hit, exploring previous sample")
            idx -= 1
        self.explore(idx, callback)


class BaseImageDataset(BaseDataset):
    def __init__(self, root: str, split: str, transform: A.Compose):
        super().__init__(root, split, transform)
        self.images_filepaths = sorted(glob.glob(f"{str(self.root)}/images/{split}/*"))

    def load_image(self, idx: int) -> np.ndarray:
        image = np.asarray(Image.open(self.images_filepaths[idx]))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
