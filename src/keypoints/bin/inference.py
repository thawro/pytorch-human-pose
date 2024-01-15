import torch
from torch import nn, Tensor
from src.utils.model import seed_everything

from src.keypoints.bin.utils import create_model
from src.keypoints.bin.config import create_config
from src.utils.video import process_video
from src.keypoints.transforms import MPPEKeypointsTransform
from src.keypoints.results import InferenceMPPEKeypointsResult
from src.keypoints.visualization import plot_heatmaps, plot_connections
import cv2
import numpy as np
import albumentations as A
from src.utils.config import RESULTS_PATH, DS_ROOT
from functools import partial
from src.base.datasets import BaseImageDataset
from src.keypoints.datasets import coco_limbs, mpii_limbs
from src.utils.image import make_grid
from albumentations.pytorch.transforms import ToTensorV2
import torch
import math


class MPPEInferenceKeypointsModel(nn.Module):
    def __init__(self, net: nn.Module, device: str):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        size = 480
        self.transform = MPPEKeypointsTransform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            symmetric_keypoints=None,
            out_size=(size, size),
        )

    def prepare_input(self, frame: np.ndarray) -> Tensor:
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255

        h, w = frame.shape[:2]
        aspect_ratio = h / w
        size = max(h, w)
        mode = "shortest"  # shortest/longest max size
        compared_size = h if mode == "shortest" else w
        divider = 32
        if size == compared_size:
            new_h = 512
            new_w = int(new_h / aspect_ratio)
            pad_y = 0
            pad_x = math.ceil(new_w / divider) * divider - new_w
        else:
            new_w = 512
            new_h = int(new_w * aspect_ratio)
            pad_x = 0
            pad_y = math.ceil(new_h / divider) * divider - new_h

        scale = h / new_h
        pad_top = math.ceil(pad_y / 2)
        pad_bot = pad_y - pad_top
        pad_left = math.ceil(pad_x / 2)
        pad_right = pad_x - pad_left
        pad = [pad_top, pad_bot, pad_left, pad_right]

        new_frame = cv2.resize(frame, (new_w, new_h))
        new_frame = cv2.copyMakeBorder(new_frame, *pad, cv2.BORDER_CONSTANT, value=0)
        new_frame = (new_frame - mean) / std
        new_frame = torch.from_numpy(new_frame).permute(2, 0, 1).float()
        return new_frame, scale, pad

    def __call__(self, frame: np.ndarray) -> InferenceMPPEKeypointsResult:
        x, scale, pad = self.prepare_input(frame)
        x = x.unsqueeze(0).to(self.device)
        image = self.transform.inverse_preprocessing(x.cpu().numpy()[0])
        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.net(x)
        return InferenceMPPEKeypointsResult.from_preds(
            frame,
            image,
            scale,
            pad,
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            max_num_people=20,
            det_thr=0.2,
            tag_thr=1.0,
        )


def processing_fn(
    model: MPPEInferenceKeypointsModel, limbs: list[tuple[int, int]], frame: np.ndarray
) -> dict:
    with torch.no_grad():
        result = model(frame)
    pred_kpts_heatmaps = plot_heatmaps(
        result.image, result.pred_kpts_heatmaps, clip_0_1=True, minmax=False
    )

    pred_tags_heatmaps_0 = plot_heatmaps(
        result.image, result.pred_tags_heatmaps[..., 0], clip_0_1=False, minmax=True
    )
    pred_tags_heatmaps_1 = plot_heatmaps(
        result.image, result.pred_tags_heatmaps[..., 1], clip_0_1=False, minmax=True
    )

    image = plot_connections(
        result.image.copy(),
        result.pred_keypoints,
        result.pred_scores,
        limbs,
        thr=0.2,
    )
    pred_kpts_heatmaps.insert(0, image)
    pred_tags_heatmaps_0.insert(0, image)
    pred_tags_heatmaps_1.insert(0, image)

    pred_kpts_grid = make_grid(pred_kpts_heatmaps, nrows=2, pad=5)
    pred_tags_grid_0 = make_grid(pred_tags_heatmaps_0, nrows=2, pad=5)
    pred_tags_grid_1 = make_grid(pred_tags_heatmaps_1, nrows=2, pad=5)

    preds_grid = np.concatenate(
        [pred_kpts_grid, pred_tags_grid_0, pred_tags_grid_1], axis=0
    )
    preds_grid = cv2.resize(preds_grid, dsize=(0, 0), fx=0.4, fy=0.4)

    cv2.imshow("grid", cv2.cvtColor(preds_grid, cv2.COLOR_RGB2BGR))

    raw_image = plot_connections(
        result.raw_image.copy(),
        result.scaled_pred_keypoints,
        result.pred_scores,
        limbs,
        thr=0.2,
    )
    cv2.imshow("Pred", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))

    return {}


def main() -> None:
    device_id = 1
    device = f"cuda:{device_id}"
    dataset = "COCO"

    if dataset == "COCO":
        ckpt_path = str(
            RESULTS_PATH
            / "test/01-12_15:17__sigmoid_MPPE_COCO_HigherHRNet/01-14_20:44/checkpoints/last.pt"
        )
    else:
        ckpt_path = str(
            RESULTS_PATH
            / "test/01-10_13:21__sigmoid_MPPE_MPII_HigherHRNet/01-11_09:10/checkpoints/last.pt"
        )
    cfg = create_config(
        dataset,
        "MPPE",
        "HigherHRNet",
        device_id,
        ckpt_path=ckpt_path,
        distributed=False,
    )

    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    net = create_model(cfg).net
    model = MPPEInferenceKeypointsModel(net, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    for key in list(ckpt.keys()):
        ckpt[key.replace("module.", "")] = ckpt[key]
        ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()

    ds = BaseImageDataset(
        root=str(DS_ROOT / f"{dataset}/HumanPose"),
        split="val",
        transform=None,
    )
    limbs = coco_limbs if dataset == "COCO" else mpii_limbs
    ds.perform_inference(partial(processing_fn, model=model, limbs=limbs))
    # process_video(partial(processing_fn, model=model), filename=0)


if __name__ == "__main__":
    main()
