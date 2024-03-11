from functools import partial

import cv2
import numpy as np

from src.base.datasets import InferenceVideoDataset
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.keypoints.model import InferenceKeypointsModel
from src.keypoints.visualization import plot_connections
from src.logger.pylogger import log
from src.utils.config import RESULTS_PATH, YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import seed_everything


def prepare_inference_config(cfg_path: str, ckpt_path: str) -> KeypointsConfig:
    cfg = load_yaml(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg = KeypointsConfig.from_dict(cfg)
    # cfg.inference.input_size = 512
    log.info("Inference config prepared.")
    log.info(f"Inference settings:\n{cfg.inference}")
    return cfg


def dataset_processing_fn(
    model: InferenceKeypointsModel,
    image: np.ndarray,
    annot: list[dict] | None = None,
) -> dict:
    result = model(image, annot)
    print("=" * 100)
    plots = result.plot()
    hm_plot = cv2.cvtColor(cv2.resize(plots["heatmaps"], (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR)
    connections_plot = cv2.cvtColor(plots["connections"], cv2.COLOR_RGB2BGR)
    ae_plot = cv2.cvtColor(plots["associative_embedding"], cv2.COLOR_RGB2BGR)
    cv2.imshow("Heatmaps", hm_plot)
    cv2.imshow("Associative Embeddings", ae_plot)
    cv2.imshow("Joints", connections_plot)
    return {}


def dataset_inference(model: InferenceKeypointsModel, cfg: KeypointsConfig):
    ds_cfg = cfg.dataloader.val_ds
    ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)
    callback = partial(dataset_processing_fn, model=model)
    ds.perform_inference(callback, idx=0, load_annot=True)


def video_processing_fn(model: InferenceKeypointsModel, image: np.ndarray) -> dict:
    result = model(image, None)

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
    connections_plot = cv2.resize(connections_plot, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("Joints", connections_plot)
    return {"out_frame": connections_plot}


def video_inference(model: InferenceKeypointsModel, filepath: str):
    ds = InferenceVideoDataset(filepath=filepath, out_filepath=None, start_frame=0, num_frames=100)
    callback = partial(video_processing_fn, model=model)
    ds.run(callback)


def main() -> None:
    seed_everything(42)
    ckpt_path = (
        f"{RESULTS_PATH}/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35/checkpoints/best.pt"
    )
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")
    cfg = prepare_inference_config(cfg_path, ckpt_path)
    model = cfg.create_inference_model("cuda:0")
    # dataset_inference(model, cfg)
    video_inference(model, "data/examples/simple_2.mp4")


if __name__ == "__main__":
    main()
