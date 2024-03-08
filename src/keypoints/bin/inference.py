from functools import partial

import cv2
import numpy as np
import torch
import torchvision
from torch import Tensor, nn

from src.base.datasets import BaseImageDataset
from src.base.transforms.utils import resize_align_multi_scale
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import coco_limbs
from src.keypoints.results import InferenceKeypointsResult
from src.logger.pylogger import log
from src.utils.config import DS_ROOT, YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import seed_everything

coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class InferenceKeypointsModel:
    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
        use_flip: bool = False,
        device: str = "cuda:1",
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.input_size = 512
        self.limbs = coco_limbs
        self.use_flip = use_flip

    def prepare_input(
        self, image: np.ndarray
    ) -> tuple[Tensor, tuple[int, int], tuple[float, float]]:
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

    def __call__(self, image: np.ndarray, annot) -> InferenceKeypointsResult:
        x, center, scale = self.prepare_input(image)

        kpts_heatmaps, tags_heatmaps = self.net(x)
        if self.use_flip:
            flip_kpts_heatmaps, flip_tags_heatmaps = self.net(torch.flip(x, [3]))
            for i in range(len(kpts_heatmaps)):
                pred_hms = kpts_heatmaps[i]
                flip_pred_hms = torch.flip(flip_kpts_heatmaps[i], [3])
                kpts_heatmaps[i] = (pred_hms + flip_pred_hms[:, coco_flip_index]) / 2
            tags_heatmaps = [
                tags_heatmaps,
                torch.flip(flip_tags_heatmaps, [3])[:, coco_flip_index],
            ]
        else:
            tags_heatmaps = [tags_heatmaps]

        input_image = x[0]
        return InferenceKeypointsResult.from_preds(
            input_image,
            image,
            scale,
            center,
            kpts_heatmaps,
            tags_heatmaps,
            self.limbs,
            max_num_people=30,
            det_thr=self.det_thr,
            tag_thr=self.tag_thr,
            annot=annot,
        )


def processing_fn(
    model: InferenceKeypointsModel,
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


def load_model(cfg_path: str, ckpt_path: str, device_id: int = 0) -> InferenceKeypointsModel:
    cfg = load_yaml(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg["setup"]["ckpt_path"] = ckpt_path

    cfg = KeypointsConfig.from_dict(cfg)

    device = f"cuda:{device_id}"

    net = cfg.create_net()
    model = InferenceKeypointsModel(
        net,
        device=device,
        det_thr=0.1,
        tag_thr=1.0,
        use_flip=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = parse_checkpoint(ckpt["module"]["model"])
    model.net.load_state_dict(ckpt)
    model.net.eval()
    log.info(f"Loaded model from {ckpt_path}")
    return model


def _set_paths(self: BaseImageDataset):
    log.info("setting")
    import glob

    images_filepaths = glob.glob(f"{self.root}/images/{self.split}/*")
    annots_filepaths = [
        path.replace("images/", "annots/").replace(".jpg", ".yaml") for path in images_filepaths
    ]
    self.images_filepaths = np.array(sorted(images_filepaths), dtype=np.str_)
    self.annots_filepaths = np.array(sorted(annots_filepaths), dtype=np.str_)


def main() -> None:
    seed_everything(42)
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-02_08:35__COCO_HigherHRNet/03-02_08:35/checkpoints/best.pt"
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35/checkpoints/best.pt"
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")

    model = load_model(cfg_path, ckpt_path, device_id=1)

    BaseImageDataset._set_paths = _set_paths
    ds = BaseImageDataset(root=str(DS_ROOT / "COCO/HumanPose"), split="val")
    ds._set_paths()
    ds.perform_inference(partial(processing_fn, model=model))


if __name__ == "__main__":
    main()
