"""Dataset classes"""

from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import glob
from typing import Callable, Any
from src.utils.files import load_yaml
from src.base.transforms.transforms import ImageTransform


class BaseDataset(Dataset):
    root: Path

    def __init__(self, root: str, split: str, transform: ImageTransform | None = None):
        self.transform = transform
        self.split = split
        self.root = Path(root)
        self.is_train = split == "train"

    def plot(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def explore(
        self, idx: int = 0, callback: Callable[[int], Any] | None = None, **kwargs
    ):
        if callback is not None:
            callback(idx)
        image = self.plot(idx, **kwargs)
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
        self.explore(idx, callback, **kwargs)


class BaseImageDataset(BaseDataset):
    def __init__(self, root: str, split: str, transform: ImageTransform | None = None):
        super().__init__(root, split, transform)
        self.images_filepaths, self.annots_filepaths = (
            self.get_images_annots_filepaths()
        )

    def get_images_annots_filepaths(self) -> tuple[np.ndarray, np.ndarray]:
        images_filepaths = sorted(glob.glob(f"{str(self.root)}/images/{self.split}/*"))
        if len(images_filepaths) == 0:
            images_filepaths = sorted(glob.glob(f"{str(self.root)}/{self.split}/*"))
        annots_filepaths = [
            path.replace("images/", "annots/").replace(".jpg", ".yaml")
            for path in images_filepaths
        ]
        images_filepaths = np.array(images_filepaths).astype(np.string_)
        annots_filepaths = np.array(annots_filepaths).astype(np.string_)

        return images_filepaths, annots_filepaths

    def __len__(self) -> int:
        return len(self.images_filepaths)

    # TODO: not all datasets have this form
    def load_image(self, idx: int) -> np.ndarray:
        image = np.asarray(Image.open(self.images_filepaths[idx]))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def load_annot(self, idx: int) -> dict:
        annot_path = self.annots_filepaths[idx]
        try:
            return load_yaml(annot_path)
        except UnicodeDecodeError:
            return load_txt(annot_path)

    def perform_inference(
        self,
        callback: Callable[[np.ndarray], Any],
        idx: int = 0,
    ):
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        callback(frame=image, annot=annot)
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
        self.perform_inference(callback, idx)
