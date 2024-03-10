from functools import partial

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn

from src.classification.config import ClassificationConfig
from src.classification.datasets import ImagenetClassificationDataset
from src.classification.results import InferenceClassificationResult
from src.logger.pylogger import log
from src.utils.config import YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import parse_checkpoint, seed_everything


class InferenceClassificationModel:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(int(224 / 0.875)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        net: nn.Module,
        idx2label: dict[int, str],
        device: str = "cuda:0",
    ):
        self.net = net.to(device)
        self.device = device
        self.idx2label = idx2label

    def prepare_input(self, image: np.ndarray) -> Tensor:
        x = self.transform(image)
        x = x.unsqueeze(0).to(self.device)
        return x

    def __call__(
        self, raw_image: np.ndarray, target_label: int | str | None = None
    ) -> InferenceClassificationResult:
        x = self.prepare_input(raw_image)
        with torch.no_grad():
            logits = self.net(x)

        return InferenceClassificationResult.from_preds(
            raw_image=raw_image,
            model_input_image=x[0],
            logits=logits[0],
            target_label=target_label,
            idx2label=self.idx2label,
        )


def processing_fn(
    model: InferenceClassificationModel, image: np.ndarray, annot: int | str | None = None
):
    result = model(image, annot)
    print("=" * 100)
    plots = result.plot()
    top_preds_plot = plots["top_preds"]
    cv2.imshow(
        "Top predictions",
        cv2.cvtColor(top_preds_plot, cv2.COLOR_RGB2BGR),
    )


def prepare_inference_config(cfg_path: str, ckpt_path: str) -> ClassificationConfig:
    cfg = load_yaml(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg = ClassificationConfig.from_dict(cfg)
    log.info("Inference config prepared.")
    log.info(f"Inference settings:\n{cfg.inference}")
    return cfg


def load_model(
    cfg: ClassificationConfig,
    ckpt_path: str,
    idx2label: dict[int, str],
) -> InferenceClassificationModel:
    device_id = 0
    device = f"cuda:{device_id}"

    net = cfg.create_net()
    model = InferenceClassificationModel(net, idx2label=idx2label, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    ckpt = parse_checkpoint(ckpt)
    model.net.load_state_dict(ckpt)
    model.net.eval()
    log.info(f"Loaded model from {ckpt_path}")
    return model


def main() -> None:
    seed_everything(42)
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-15_10:12___imagenet_HRNet/02-19_09:14/checkpoints/best.pt"
    cfg_path = str(YAML_EXP_PATH / "classification" / "hrnet_32.yaml")
    cfg = prepare_inference_config(cfg_path, ckpt_path)

    ds_cfg = cfg.dataloader.val_ds
    ds = ImagenetClassificationDataset(root=ds_cfg.root, split=ds_cfg.split)

    model = load_model(cfg, ckpt_path, idx2label=ds.idx2label)

    ds.perform_inference(partial(processing_fn, model=model), load_annot=True)


if __name__ == "__main__":
    main()
