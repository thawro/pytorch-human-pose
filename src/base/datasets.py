"""Dataset classes"""

import glob
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils.files import load_yaml
from src.utils.image import make_grid


class ExplorerDataset:
    def plot(self, idx: int, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def plot_examples(self, idxs: list[int], **kwargs) -> np.ndarray:
        samples_plots = [self.plot(idx, **kwargs) for idx in idxs]
        grid = make_grid(samples_plots, nrows=len(samples_plots), pad=20)
        return grid

    def explore(self, idx: int = 0, callback: Callable[[int], Any] | None = None, **kwargs):
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


class InferenceDataset:
    def load_image(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def load_annot(self, idx: int) -> dict:
        raise NotImplementedError()

    def perform_inference(self, callback: Callable[[np.ndarray], Any], idx: int = 0):
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


class BaseImageDataset(Dataset, ExplorerDataset, InferenceDataset):
    images_filepaths: np.ndarray
    annots_filepaths: np.ndarray

    def __init__(self, root: str, split: str, transform: Callable | None = None):
        self.transform = transform
        self.split = split
        self.root = root
        self.is_train = split == "train"

    def _set_paths(self):
        # set images_filepaths and annots_filepaths
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.annots_filepaths)

    def load_image(self, idx: int) -> np.ndarray:
        return np.asarray(Image.open(self.images_filepaths[idx]).convert("RGB"))

    def load_annot(self, idx: int) -> dict:
        return load_yaml(self.annots_filepaths[idx])
