from functools import partial

import cv2
import numpy as np

from src.classification.config import ClassificationConfig
from src.classification.datasets import ImagenetClassificationDataset
from src.classification.model import InferenceClassificationModel
from src.logger.pylogger import log
from src.utils.config import YAML_EXP_PATH
from src.utils.files import load_yaml
from src.utils.model import seed_everything


def prepare_inference_config(cfg_path: str, ckpt_path: str) -> ClassificationConfig:
    cfg = load_yaml(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg = ClassificationConfig.from_dict(cfg)
    log.info("Inference config prepared.")
    log.info(f"Inference settings:\n{cfg.inference}")
    return cfg


def main() -> None:
    seed_everything(42)
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-15_10:12___imagenet_HRNet/02-19_09:14/checkpoints/best.pt"
    cfg_path = str(YAML_EXP_PATH / "classification" / "hrnet_32.yaml")
    cfg = prepare_inference_config(cfg_path, ckpt_path)

    ds_cfg = cfg.dataloader.val_ds
    ds = ImagenetClassificationDataset(root=ds_cfg.root, split=ds_cfg.split)

    model = cfg.create_inference_model(idx2label=ds.idx2label, device="cuda:0")
    ds.perform_inference(model=model, idx=0, load_annot=False)


if __name__ == "__main__":
    main()
