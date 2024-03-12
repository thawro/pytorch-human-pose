"""Dataset class for Video inference"""

from dataclasses import dataclass, fields
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm
from typing_extensions import Protocol

from src.logger.pylogger import log
from src.utils.image import put_txt

from .base import KeyBinds


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


@dataclass
class VideoProcessingResult:
    speed_ms: dict[str, int]
    model_input_shape: tuple[int, int]
    out_frame: np.ndarray | None = None
    idx: int | None = None

    @property
    def stats(self) -> dict[str, Any]:
        return {"idx": self.idx, "speed": self.speed_ms}


class VideoProcessingCallback(Protocol):
    def __call__(self, image: np.ndarray) -> VideoProcessingResult: ...


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
        if num_frames == -1:
            num_frames = cap_props.num_frames
        self.num_frames = min(num_frames, cap_props.num_frames)
        self.is_paused = False
        self.is_tab = False
        self.arrow_hit = False
        self.cap = cap
        self.cap_props = cap_props
        self.results = {"filepath": filepath, **cap_props.to_dict(), "frames": []}
        self.out_filepath = out_filepath
        self.out_cap = None

    def _set_out_cap(self, height: int, width: int):
        if self.out_filepath is None or self.out_cap is not None:
            return
        ext = self.out_filepath.split(".")[-1]
        # NOTE: if writing video doesn't work try installing: `sudo apt-get install ffmpeg x264 libx264-dev`
        codecs = {
            "mp4": "MP4V",  # try one of: h264, x264, mp4v (it is platform dependent)
            "avi": "MJPG",
        }
        codec = codecs.get(ext, "MJPG")
        try:
            self.out_cap = cv2.VideoWriter(
                self.out_filepath,
                cv2.VideoWriter_fourcc(*codec),
                self.cap_props.fps,
                (width, height),
            )
            log.info(
                "Output VideoWritter initialized successfully. "
                f"The output frames will be saved to {self.out_filepath}"
            )
        except Exception as e:
            log.exception(e)
            log.warning(
                "There was an error during `out_cap` (`cv2.VideoWritter`) initialization. "
                "The inference will continue without writing the out_frames to the output video file."
            )
            self.out_filepath = None
            # uncomment that if you preffer to raise the Error instead
            # raise e

    def on_start(self):
        log.info(f"Started processing {self.filepath} video file")

    def on_end(self):
        self.cap.release()
        log.info(f"Released {self.filepath} VideoCapture")
        if self.out_filepath is not None and self.out_cap is not None:
            self.out_cap.release()
            log.info(f"Released {self.out_filepath} VideoWritter")
        cv2.destroyAllWindows()
        log.info(f"Ended processing {self.filepath} video file")

    @property
    def should_process(self) -> bool:
        return self.idx < self.num_frames or self.num_frames < 0

    def move_by_n_frames(self, n_frames: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx + n_frames)
        self.idx += n_frames
        self.pbar.update(n_frames)

    def run(self, callback: VideoProcessingCallback):
        self.on_start()

        self.pbar = tqdm(total=self.num_frames, desc=f"Processing video file ({self.filepath})")
        while self.cap.isOpened() and self.should_process:
            try:
                self.process(callback)
                key = cv2.waitKey(1)
                if key in [KeyBinds.ESCAPE]:
                    e = KeyboardInterrupt("Escape key hit. Interrupting")
                    log.exception(e)
                    raise e
                elif key in [KeyBinds.SPACE]:
                    self.is_paused = not self.is_paused
                elif key in [KeyBinds.TAB]:
                    self.is_tab = not self.is_tab
                elif key in [KeyBinds.LEFT_ARROW, KeyBinds.RIGHT_ARROW]:
                    base_move = 1  # there is always a bonus cap.read() in process method
                    n_frames = -1 if key == KeyBinds.LEFT_ARROW else 1
                    self.move_by_n_frames(n_frames - base_move)
                    self.arrow_hit = True
                    self.is_paused = True
            except (StopIteration, KeyboardInterrupt):
                break
        self.on_end()

    def draw_labels(self, image: np.ndarray, result: VideoProcessingResult):
        input_h, input_w = result.model_input_shape
        idx_label = f"{self.idx} / {self.num_frames}"
        shape_label = f"Model input shape: {input_h} x {input_w}"
        speed_labels = ["Latency [ms]: "] + [
            f"  {name}: {value}" for name, value in result.speed_ms.items()
        ]
        labels = [idx_label, shape_label, *speed_labels]
        put_txt(image, labels, loc="tl", alpha=0.6, font_scale=0.5)

    def process(self, callback: VideoProcessingCallback):
        if (self.is_paused and self.arrow_hit) or not self.is_paused or self.arrow_hit:
            success, frame = self.cap.read()
            if not success:
                raise StopIteration("Couldn't read next frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = callback(image=frame)
            result.idx = self.idx
            out_frame = result.out_frame
            if out_frame is not None:
                out_h, out_w = out_frame.shape[:2]
                self._set_out_cap(out_h, out_w)
                self.draw_labels(out_frame, result)

                if self.out_filepath is not None and self.out_cap is not None:
                    self.out_cap.write(out_frame)
                if self.is_tab:
                    put_txt(out_frame, list(KeyBinds.key2info.values()), loc="bl", alpha=0.6)
                cv2.imshow("Out frame", out_frame)

            self.results["frames"].append(result.stats)
            self.pbar.update(1)
            self.idx += 1
        else:
            return
        self.arrow_hit = False
