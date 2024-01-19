import torch
from torch import nn, Tensor
from src.utils.model import seed_everything

from src.keypoints.bin.utils import create_model
from src.keypoints.bin.config import create_config
from src.keypoints.config import Config
from src.keypoints.transforms import MPPEKeypointsTransform
from src.keypoints.results import InferenceMPPEKeypointsResult
from src.keypoints.datasets import coco_symmetric_labels
import cv2
import numpy as np
from src.utils.config import RESULTS_PATH, DS_ROOT
from functools import partial
from src.base.datasets import BaseImageDataset
from src.keypoints.datasets import coco_limbs, mpii_limbs
import torch
import math


class MPPEInferenceKeypointsModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
        device: str = "cuda:1",
        limbs: list[tuple[int, int]] = coco_limbs,
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        size = 512
        self.limbs = limbs
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
        mode = "longest"
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

    def __call__(self, frame: np.ndarray, annot) -> InferenceMPPEKeypointsResult:
        x, scale, pad = self.prepare_input(frame)
        x = x.unsqueeze(0).to(self.device)
        input_image = self.transform.inverse_preprocessing(x.cpu().numpy()[0])
        x_fliplr = torch.flip(x, (3,))
        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.net(x)
        # stages_pred_kpts_heatmaps_flip, stages_pred_tags_heatmaps_flip = self.net(
        #     x_fliplr
        # )
        # for i, (kpts_hms_flipped, tags_hms_flipped) in enumerate(
        #     zip(stages_pred_kpts_heatmaps_flip, stages_pred_tags_heatmaps_flip)
        # ):
        #     kpts_hms = kpts_hms_flipped.flip((3,))[0][coco_symmetric_labels].unsqueeze(
        #         0
        #     )
        #     tags_hms = tags_hms_flipped.flip((3,))[0][coco_symmetric_labels]

        #     stages_pred_kpts_heatmaps[i] = (stages_pred_kpts_heatmaps[i] + kpts_hms) / 2
        #     stages_pred_tags_heatmaps[i] = torch.stack(
        #         [stages_pred_tags_heatmaps[i][0], tags_hms], dim=0
        #     )

        return InferenceMPPEKeypointsResult.from_preds(
            annot,
            input_image,
            frame,
            scale,
            pad,
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            self.limbs,
            max_num_people=20,
            det_thr=self.det_thr,
            tag_thr=self.tag_thr,
        )


def processing_fn(
    model: MPPEInferenceKeypointsModel,
    frame: np.ndarray,
    annot,
) -> dict:
    with torch.no_grad():
        result = model(frame, annot)
    final_plot, raw_image = result.plot()
    cv2.imshow("grid", cv2.cvtColor(final_plot, cv2.COLOR_RGB2BGR))
    cv2.imshow("Pred", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
    return {}


def load_model(dataset: str = "COCO"):
    device_id = 1
    device = f"cuda:{device_id}"
    limbs = coco_limbs if dataset == "COCO" else mpii_limbs
    if dataset == "COCO":
        ckpt_path = str(
            RESULTS_PATH
            / "test/01-12_15:17__sigmoid_MPPE_COCO_HigherHRNet/01-14_20:44/checkpoints/last.pt"
        )
        ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/01-17_16:04__sigmoid_MPPE_COCO_HigherHRNet/01-18_11:10/checkpoints/best.pt"
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
    model = MPPEInferenceKeypointsModel(net, device=device, limbs=limbs, det_thr=0.1)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    for key in list(ckpt.keys()):
        ckpt[key.replace("module.", "")] = ckpt[key]
        ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def main() -> None:
    dataset = "COCO"
    model = load_model(dataset)
    ds = BaseImageDataset(
        root=str(DS_ROOT / f"{dataset}/HumanPose"),
        split="val",
        transform=None,
    )

    ds.perform_inference(partial(processing_fn, model=model))
    # process_video(partial(processing_fn, model=model), filename=0)


if __name__ == "__main__":
    main()
