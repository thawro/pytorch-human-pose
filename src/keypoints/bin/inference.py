from functools import partial
from pathlib import Path

import cv2
import numpy as np

from src.base.bin.inference import prepare_inference_config
from src.base.datasets import InferenceVideoDataset
from src.base.datasets.video import VideoProcessingResult
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.keypoints.model import InferenceKeypointsModel
from src.keypoints.visualization import plot_connections
from src.utils.config import YAML_EXP_PATH
from src.utils.image import resize_with_aspect_ratio
from src.utils.utils import elapsed_timer


def dataset_inference(model: InferenceKeypointsModel, cfg: KeypointsConfig):
    ds_cfg = cfg.dataloader.val_ds
    ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)
    ds.perform_inference(model=model, idx=0, load_annot=False)


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
    path_parts = filepath.split("/")
    in_filepath_dir = "/".join(path_parts[:-1])
    filename, ext = path_parts[-1].split(".")
    out_filepath_dir = f"{in_filepath_dir}/out"
    Path(out_filepath_dir).mkdir(exist_ok=True, parents=True)
    out_filepath = f"{out_filepath_dir}/{filename}.{ext}"
    ds = InferenceVideoDataset(
        filepath=filepath, out_filepath=out_filepath, start_frame=0, num_frames=-1
    )
    callback = partial(video_processing_fn, model=model)
    ds.run(callback)


def main() -> None:
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")
    cfg: KeypointsConfig = prepare_inference_config(cfg_path, KeypointsConfig)
    model = cfg.create_inference_model(device="cuda:0")
    # dataset_inference(model, cfg)
    video_inference(model, "data/examples/simple_2.mp4")


if __name__ == "__main__":
    main()
