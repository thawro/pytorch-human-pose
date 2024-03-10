"""Base Dataset classes"""

from dataclasses import dataclass, fields
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing_extensions import Protocol

from src.logger.pylogger import log
from src.utils.image import make_grid


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
    def __call__(self, image: np.ndarray, annot: Any) -> Any: ...


class InferenceDataset:
    def load_image(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def load_annot(self, idx: int) -> dict:
        raise NotImplementedError()

    def perform_inference(
        self, callback: PerformInferenceCallback, idx: int = 0, load_annot: bool = False
    ):
        image = self.load_image(idx)

        annot = self.load_annot(idx) if load_annot else None
        callback(image=image, annot=annot)
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
        self.perform_inference(callback, idx, load_annot)


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


@dataclass
class CapProps:
    height: int
    width: int
    fps: int
    start_frame: int
    num_frames: int

    def to_dict(self) -> dict:
        dct = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            dct[field_name] = field_value
        return dct


def prepare_video(
    cap: cv2.VideoCapture, start_frame: int, num_frames: int
) -> tuple[cv2.VideoCapture, CapProps]:
    # TODO add info about input and output extensions + define outptu writer codec
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_props = CapProps(
        height=height, width=width, fps=fps, start_frame=start_frame, num_frames=num_frames
    )
    return cap, cap_props


class VideoProcessingCallback(Protocol):
    def __call__(self, image: np.ndarray) -> dict | None: ...


class InferenceVideoDataset:
    def __init__(
        self,
        filepath: str,
        out_filepath: str | None = None,
        start_frame: int = 0,
        num_frames: int = -1,
    ):
        self.filepath = filepath
        cap = cv2.VideoCapture(filepath)
        cap, cap_props = prepare_video(cap, start_frame, num_frames)
        self.idx = start_frame
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.cap = cap
        self.cap_props = cap_props
        self.results = {"filepath": filepath, **cap_props.to_dict(), "frames": []}
        self.out_filepath = out_filepath
        if out_filepath is not None:
            self.out_cap = cv2.VideoWriter(
                out_filepath,
                cv2.VideoWriter_fourcc(*"MJPG"),
                cap_props.fps,
                (cap_props.width, cap_props.height),
            )

    def on_start(self):
        log.info(f"Started processing {self.filepath} video file")

    def on_end(self):
        self.cap.release()
        log.info(f"Released {self.filepath} VideoCapture")
        if self.out_filepath is not None:
            self.out_cap.release()
            log.info(f"Released {self.out_filepath} VideoWritter")
        log.info(f"Ended processing {self.filepath} video file")

    @property
    def should_process(self) -> bool:
        return self.idx < self.num_frames or self.num_frames < 0

    def run(self, callback: VideoProcessingCallback):
        self.on_start()

        pbar = tqdm(total=self.num_frames, desc=f"Processing video file ({self.filepath})")
        while self.cap.isOpened() and self.should_process:
            try:
                self.process(callback)
                key = cv2.waitKey(1)
                if key == 27:  # escape hit
                    raise KeyboardInterrupt("Escape key hit. Interrupting")
            except (StopIteration, KeyboardInterrupt):
                break
            else:
                pbar.update(1)
                self.idx += 1
        self.on_end()

    def process(self, callback: VideoProcessingCallback):
        success, frame = self.cap.read()
        if not success:
            raise StopIteration("Couldn't read next frame")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = callback(image=frame)
        if result is not None:
            result["idx"] = self.idx
            if "out_frame" in result.keys() and self.out_filepath is not None:
                out_frame = result.pop("out_frame")
                out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                self.out_cap.write(out_frame)
        self.results["frames"].append(result)
