import argparse
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from src.base.bin.inference import prepare_inference_config
from src.base.datasets import DirectoryDataset, InferenceVideoDataset
from src.base.datasets.video import VideoProcessingResult
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.keypoints.model import InferenceKeypointsModel
from src.keypoints.visualization import plot_connections
from src.logger.pylogger import log, log_breaking_point
from src.utils.config import INFERENCE_OUT_PATH, YAML_EXP_PATH
from src.utils.image import resize_with_aspect_ratio
from src.utils.utils import elapsed_timer

KPTS_INFERENCE_OUT_PATH = INFERENCE_OUT_PATH / "keypoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ClassificationHRNet Inference",
        description="Perform inference of ClassificationHRNet neural network trained on ImageNet dataset",
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=str,
        help="Inference mode. If 'val', then COCO val split is used, if 'custom', then path is used.",
        default="val",
    )
    parser.add_argument(
        "--path",
        action="store",
        type=str,
        help="Path to directory with images or to video file for inference. Used only when 'mode' is set to 'custom'",
        default=None,
    )
    args, unknown = parser.parse_known_args()
    log.info(f"Parsed args: \n{args}")
    return args


def video_processing_fn(model: InferenceKeypointsModel, image: np.ndarray) -> VideoProcessingResult:
    speed_ms = {}
    with elapsed_timer() as latency_sec:
        latency_sec()
        result = model(image, None)
        speed_ms["inference"] = int(latency_sec() * 1000)
    model_input_shape = model.model_input_shape
    # sort results by tags so visualizations are nice (same colors for persons)
    obj_ref_tags = result.kpts_tags.mean(axis=1)
    sort_idxs = np.argsort(obj_ref_tags[:, 0])
    sorted_kpts_coords = result.kpts_coords[sort_idxs]
    sorted_kpts_scores = result.kpts_scores[sort_idxs]
    # sorted_kpts_tags = result.kpts_tags[sort_idxs]

    connections_plot = plot_connections(
        result.raw_image.copy(),
        sorted_kpts_coords,
        sorted_kpts_scores,
        result.limbs,
        thr=result.det_thr,
        color_mode="limb",
        alpha=0.65,
    )

    connections_plot = cv2.cvtColor(connections_plot, cv2.COLOR_RGB2BGR)
    connections_plot = resize_with_aspect_ratio(connections_plot, height=640, width=None)
    return VideoProcessingResult(speed_ms, model_input_shape, out_frame=connections_plot, idx=None)


def video_inference(model: InferenceKeypointsModel, filepath: str):
    filename, ext = filepath.split("/")[-1].split(".")
    out_filepath_dir = KPTS_INFERENCE_OUT_PATH / "video"
    out_filepath_dir.mkdir(exist_ok=True, parents=True)
    out_filepath = f"{str(out_filepath_dir)}/{filename}_Size({model.input_size}).{ext}"
    ds = InferenceVideoDataset(
        filepath=filepath, out_filepath=out_filepath, start_frame=0, num_frames=-1
    )
    callback = partial(video_processing_fn, model=model)
    ds.run(callback)


def main() -> None:
    log_breaking_point("Starting inference", n_top=1, n_bottom=1, top_char="*", bottom_char="*")

    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")
    cfg: KeypointsConfig = prepare_inference_config(cfg_path, KeypointsConfig)
    model = cfg.create_inference_model(device="cuda:0")

    args = parse_args()

    if args.mode == "custom":
        path = args.path
        if os.path.isfile(path):
            log.info(f"Performing Video Inference ({path})")
            video_inference(model, path)
        elif os.path.isdir(path):
            log.info(f"Performing Directory Inference ({path})")
            ds = DirectoryDataset(path)
            out_dirpath = str(KPTS_INFERENCE_OUT_PATH / "custom")
            ds.perform_inference(model=model, idx=0, load_annot=False, out_dirpath=out_dirpath)
    else:
        log.info("Performing COCO val Inference")
        ds_cfg = cfg.dataloader.val_ds
        ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)
        out_dirpath = str(KPTS_INFERENCE_OUT_PATH / "val")
        ds.perform_inference(model=model, idx=0, load_annot=False, out_dirpath=out_dirpath)


if __name__ == "__main__":
    main()
