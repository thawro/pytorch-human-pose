import glob
from functools import partial

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn

from src.base.datasets import InferenceVideoDataset
from src.base.transforms.utils import resize_align_multi_scale
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import COCO_LIMBS, CocoKeypointsDataset
from src.keypoints.results import InferenceKeypointsResult
from src.keypoints.transforms import COCO_FLIP_INDEX
from src.logger.pylogger import log
from src.utils.config import RESULTS_PATH, YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import seed_everything


class InferenceKeypointsModel:
    limbs = COCO_LIMBS

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.05,
        tag_thr: float = 0.5,
        use_flip: bool = False,
        input_size: int = 512,
        max_num_people: int = 30,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.max_num_people = max_num_people
        self.input_size = input_size
        self.use_flip = use_flip

    def prepare_input(
        self, image: np.ndarray
    ) -> tuple[Tensor, tuple[int, int], tuple[float, float]]:
        image_resized, center, scale = resize_align_multi_scale(image, self.input_size, 1, 1)
        image_resized = self.transform(image_resized)
        x = image_resized.unsqueeze(0).to(self.device)
        return x, center, scale

    def __call__(self, raw_image: np.ndarray, annot: list[dict] | None) -> InferenceKeypointsResult:
        with torch.no_grad():
            x, center, scale = self.prepare_input(raw_image)
            kpts_heatmaps, tags_heatmaps = self.net(x)

            if self.use_flip:
                flip_kpts_heatmaps, flip_tags_heatmaps = self.net(torch.flip(x, [3]))
                for i in range(len(kpts_heatmaps)):
                    pred_hms = kpts_heatmaps[i]
                    flip_pred_hms = torch.flip(flip_kpts_heatmaps[i], [3])
                    kpts_heatmaps[i] = (pred_hms + flip_pred_hms[:, COCO_FLIP_INDEX]) / 2
                tags_heatmaps = [
                    tags_heatmaps,
                    torch.flip(flip_tags_heatmaps, [3])[:, COCO_FLIP_INDEX],
                ]
            else:
                tags_heatmaps = [tags_heatmaps]

        model_input_image = x[0]
        return InferenceKeypointsResult.from_preds(
            raw_image=raw_image,
            annot=annot,
            model_input_image=model_input_image,
            kpts_heatmaps=kpts_heatmaps,
            tags_heatmaps=tags_heatmaps,
            limbs=self.limbs,
            scale=scale,
            center=center,
            det_thr=self.det_thr,
            tag_thr=self.tag_thr,
            max_num_people=self.max_num_people,
        )


def load_model(cfg: KeypointsConfig, device_id: int = 0) -> InferenceKeypointsModel:
    device = f"cuda:{device_id}"
    net = cfg.create_net()
    model = InferenceKeypointsModel(
        net,
        device=device,
        det_thr=cfg.inference.det_thr,
        tag_thr=cfg.inference.tag_thr,
        use_flip=cfg.inference.use_flip,
        input_size=cfg.inference.input_size,
    )
    ckpt = torch.load(cfg.setup.ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    model.net.load_state_dict(ckpt)
    model.net.eval()
    log.info(f"Loaded model from {cfg.setup.ckpt_path}")
    return model


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


def dataset_inference(cfg: KeypointsConfig):
    model = load_model(cfg)
    ds_cfg = cfg.dataloader.val_ds
    ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)
    callback = partial(dataset_processing_fn, model=model)
    ds.perform_inference(callback, idx=0, load_annot=False)


def video_processing_fn(model: InferenceKeypointsModel, image: np.ndarray) -> dict:
    result = model(image, None)
    plots = result.plot()
    connections_plot = cv2.cvtColor(plots["connections"], cv2.COLOR_RGB2BGR)
    connections_plot = cv2.resize(connections_plot, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("Joints", connections_plot)
    return {"out_frame": connections_plot}


def video_inference(cfg: KeypointsConfig, filepath: str):
    model = load_model(cfg)
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
    # dataset_inference(cfg)
    video_inference(cfg, "data/examples/small.mp4")


if __name__ == "__main__":
    main()
