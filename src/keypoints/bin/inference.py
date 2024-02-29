from functools import partial
from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.base.datasets import BaseImageDataset
from src.base.transforms.utils import (
    affine_transform,
    get_affine_transform,
    resize_align_multi_scale,
)
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco_keypoints import coco_limbs
from src.keypoints.results import InferenceMPPEKeypointsResult
from src.logger.pylogger import log
from src.utils.config import DS_ROOT, YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import seed_everything


def transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    transform_matrix = get_affine_transform(center, scale, 0, output_size, inverse=True)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], transform_matrix)
    return target_coords


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    for person in grouped_joints:
        joints = np.zeros((person.shape[0], 3))
        joints = transform_preds(person, center, scale, heatmap_size)
        final_results.append(joints)
    return final_results


class MPPEInferenceKeypointsModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
        device: str = "cuda:1",
        ds_name: Literal["COCO", "MPII"] = "COCO",
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.input_size = 512
        self.ds_name = ds_name
        self.limbs = coco_limbs

    def prepare_input(self, image: np.ndarray) -> Tensor:
        import torchvision

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_resized, center, scale = resize_align_multi_scale(image, self.input_size, 1, 1)

        image_resized = transforms(image_resized)
        x = image_resized.unsqueeze(0).to(self.device)
        return x, center, scale

    def __call__(self, image: np.ndarray, annot) -> InferenceMPPEKeypointsResult:
        x, center, scale = self.prepare_input(image)

        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.net(x)

        input_image = x[0].permute(1, 2, 0).cpu().numpy()
        _mean = np.array([0.485, 0.456, 0.406]) * 255
        _std = np.array([0.229, 0.224, 0.225]) * 255
        input_image = (input_image * _std) + _mean
        input_image = input_image.astype(np.uint8)

        return InferenceMPPEKeypointsResult.from_preds(
            annot,
            input_image,
            image,
            scale,
            center,
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            get_final_preds,
            self.limbs,
            max_num_people=30,
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

    print("=" * 100)
    final_plot, raw_image = result.plot()
    cv2.imshow(
        "grid",
        cv2.cvtColor(cv2.resize(final_plot, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR),
    )
    cv2.imshow("Pred", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
    return {}


def parse_checkpoint(ckpt: dict) -> dict:
    # TODO: dont know why DDP saves the names as _orig_mod
    redundant_prefixes = ["module.", "_orig_mod."]
    for key in list(ckpt.keys()):
        renamed_key = str(key)
        for prefix in redundant_prefixes:
            renamed_key = renamed_key.replace(prefix, "")
        ckpt[renamed_key] = ckpt.pop(key)
    return ckpt


def load_model(cfg_path: str, ckpt_path: str) -> MPPEInferenceKeypointsModel:
    cfg = load_yaml(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg["setup"]["ckpt_path"] = ckpt_path

    # cfg["model"]["architecture"] = "OriginalHigherHRNet"
    cfg = KeypointsConfig.from_dict(cfg)

    device_id = 0
    device = f"cuda:{device_id}"

    net = cfg.create_net()
    model = MPPEInferenceKeypointsModel(
        net,
        device=device,
        ds_name=cfg.dataloader.dataset.name,
        det_thr=0.1,
        tag_thr=1.0,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    ckpt = parse_checkpoint(ckpt)
    model.load_state_dict(ckpt)
    model.eval()
    log.info(f"Loaded model from {ckpt_path}")
    return model


def main() -> None:
    seed_everything(42)
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/02-29_11:04___COCO_HigherHRNet/02-29_11:04/checkpoints/best.pt"
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")

    model = load_model(cfg_path, ckpt_path)

    ds = BaseImageDataset(root=str(DS_ROOT / f"{model.ds_name}/HumanPose"), split="val")
    ds.perform_inference(partial(processing_fn, model=model))


if __name__ == "__main__":
    main()
