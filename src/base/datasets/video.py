"""Dataset class for Video inference"""

from dataclasses import dataclass, fields

import cv2
import numpy as np
from tqdm.auto import tqdm
from typing_extensions import Protocol

from src.logger.pylogger import log


class KeyBinds:
    ESCAPE = 27
    SPACE = 32
    LEFT_ARROW = 81
    RIGHT_ARROW = 83
    DOWN_ARROW = 82
    UP_ARROW = 84

    key2str = {
        ESCAPE: "ESCAPE",
        SPACE: "SPACE",
        LEFT_ARROW: "LEFT_ARROW",
        RIGHT_ARROW: "RIGHT_ARROW",
        DOWN_ARROW: "DOWN_ARROW",
        UP_ARROW: "UP_ARROW",
    }

    @classmethod
    def to_string(cls, key: int) -> str:
        return cls.key2str[key]


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
        self.is_paused = False
        self.arrow_hit = False
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
                elif key in [KeyBinds.LEFT_ARROW, KeyBinds.RIGHT_ARROW]:
                    base_move = 1  # there is always a bonus cap.read() in process method
                    n_frames = -1 if key == KeyBinds.LEFT_ARROW else 1
                    self.move_by_n_frames(n_frames - base_move)
                    self.arrow_hit = True
                    self.is_paused = True
            except (StopIteration, KeyboardInterrupt):
                break
        self.on_end()

    def process(self, callback: VideoProcessingCallback):
        if self.is_paused and self.arrow_hit or not self.is_paused or self.arrow_hit:
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
            self.pbar.update(1)
            self.idx += 1
        else:
            return
        self.arrow_hit = False
