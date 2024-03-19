"""Base Dataset classes"""

import glob
from typing import Any, Callable

import cv2
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing_extensions import Protocol

from src.base.model import BaseInferenceModel
from src.utils.image import make_grid, resize_with_aspect_ratio


class KeyBinds:
    # change according to your system

    ESCAPE = 27
    SPACE = 32
    LEFT_ARROW = 81
    RIGHT_ARROW = 83
    DOWN_ARROW = 82
    UP_ARROW = 84
    TAB = 9

    key2info = {
        ESCAPE: "Escape - close",
        SPACE: "Space - pause",
        LEFT_ARROW: "Left Arrow - move to previous frame",
        RIGHT_ARROW: "Right Arrow - move to next frame",
        TAB: "Tab - open/close navigation bar",
        # DOWN_ARROW: "DOWN_ARROW",
        # UP_ARROW: "UP_ARROW",
    }

    @classmethod
    def to_info(cls, key: int) -> str:
        return cls.key2info[key]


class ExploreCallback(Protocol):
    def __call__(self, idx: int) -> Any: ...


class ExplorerDataset:
    def plot(self, idx: int, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def plot_examples(self, idxs: list[int], **kwargs) -> np.ndarray:
        samples_plots = [self.plot(idx, **kwargs) for idx in idxs]
        grid = make_grid(samples_plots, nrows=len(samples_plots), pad=20)
        return grid

    def explore(self, idx: int = 0, callback: ExploreCallback | None = None, **kwargs):
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


class PerformInferenceCallback(Protocol):
    def __call__(self, model: BaseInferenceModel, image: np.ndarray, annot: Any) -> Any: ...


def inference_callback(
    model: BaseInferenceModel,
    image: np.ndarray,
    annot: list[dict] | None = None,
):
    result = model(image, annot)
    print("=" * 100)
    plots = result.plot()
    for name, plot in plots.items():
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        plot = resize_with_aspect_ratio(plot, height=512, width=None)
        cv2.imshow(name.replace("_", " ").title(), plot)


class InferenceDataset:
    def load_image(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def load_annot(self, idx: int) -> dict:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def perform_inference(
        self,
        model: BaseInferenceModel,
        callback: PerformInferenceCallback = inference_callback,
        idx: int = 0,
        load_annot: bool = False,
    ):
        num_samples = len(self)
        if idx == -1:
            idx = num_samples - 1
        if idx == num_samples:
            idx = 0
        print(f"Inference for sample: {idx}/{num_samples-1}")
        image = self.load_image(idx)

        annot = self.load_annot(idx) if load_annot else None
        callback(model=model, image=image, annot=annot)
        key = cv2.waitKey(0)

        if key in [KeyBinds.ESCAPE]:  # ESC pressed
            print("Escape hit, closing")
            cv2.destroyAllWindows()
            return
        elif key in [KeyBinds.SPACE, KeyBinds.RIGHT_ARROW]:
            idx += 1
        elif key in [KeyBinds.LEFT_ARROW]:
            idx -= 1
        self.perform_inference(model, callback, idx, load_annot)


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
        return len(self.images_filepaths)

    def load_image(self, idx: int) -> np.ndarray:
        return np.array(Image.open(self.images_filepaths[idx]).convert("RGB"))

    def load_annot(self, idx: int) -> Any:
        raise NotImplementedError()


class DirectoryDataset(Dataset, InferenceDataset):
    def __init__(self, dirpath: str):
        self.dirpath = dirpath
        self.images_filepaths = natsort.natsorted(glob.glob(f"{dirpath}/*"))

    def _set_paths(self):
        # set images_filepaths and annots_filepaths
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.images_filepaths)

    def load_image(self, idx: int) -> np.ndarray:
        return np.array(Image.open(self.images_filepaths[idx]).convert("RGB"))

    def load_annot(self, idx: int) -> Any:
        return None
