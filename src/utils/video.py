from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2
from typing import Callable, Any
from collections import defaultdict
import torch


def save_frames_to_video(frames: list[np.ndarray], fps: int, filepath: str):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filepath, fps=fps)


def record_webcam_to_mp4(filename: str = "video.mp4"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow("Frame", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename)

    cap.release()
    cv2.destroyAllWindows()


def get_video_size(filename: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(filename)
    w = int(cap.get(3))
    h = int(cap.get(4))
    cap.release()
    cv2.destroyAllWindows()
    return w, h


def process_video(
    processing_fn: Callable[[np.ndarray], dict[str, Any]],
    prev_frames: torch.Tensor,
    filename: str | int = "video.mp4",
    start_frame: int = 0,
    end_frame: int = -1,
    verbose: bool = False,
) -> dict[str, list[Any]]:
    cap = cv2.VideoCapture(filename)
    if end_frame == -1:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = defaultdict(lambda: [], {})
    count = 0
    while True:
        count += 1
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if verbose:
            print(count)
        if count < start_frame:
            continue
        if count == end_frame:
            break
        if ret:
            result, prev_frames = processing_fn(frame=frame_rgb, prev_frames=prev_frames)
            for name, out in result.items():
                results[name].append(out)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    return results
