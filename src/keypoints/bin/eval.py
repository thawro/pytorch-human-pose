import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.keypoints.bin.inference import MPPEInferenceKeypointsModel, load_model
from src.utils.config import DS_ROOT, YAML_EXP_PATH
from src.utils.files import load_yaml, save_json
from src.utils.model import seed_everything


def evaluate_dataset(dataset: BaseImageDataset, model: MPPEInferenceKeypointsModel):
    filepaths = dataset.images_filepaths.tolist()
    n_examples = len(dataset)
    # n_examples = 100
    results = []
    with torch.no_grad():
        for idx in tqdm(range(n_examples)):
            image_path = filepaths[idx].decode("utf-8")
            annot_path = image_path.replace(".jpg", ".yaml").replace("images/", "annots/")
            image = np.asarray(Image.open(image_path))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if os.path.isfile(annot_path):
                annot = load_yaml(annot_path)
                image_id = int(
                    annot["filename"].replace(".jpg", "").replace("_valid", "").lstrip("0")
                )
            else:
                image_id = int(Path(image_path).stem.lstrip("0"))
            result = model(image, None)
            pred_kpts = result.pred_keypoints

            pred_scores = result.pred_obj_scores

            num_obj = len(pred_scores)

            for i in range(num_obj):
                kpts = pred_kpts[i]
                scores = pred_scores[i]
                num_kpts = len(kpts)

                coco_kpts = np.zeros((num_kpts * 3,))
                coco_kpts[::3] = kpts[:, 0]  # x
                coco_kpts[1::3] = kpts[:, 1]  # y
                coco_kpts[2::3] = 1  # v

                coco_result = {
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": coco_kpts.tolist(),
                    "score": scores.mean().item(),
                }
                results.append(coco_result)
    return results


def main() -> None:
    seed_everything(42)
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")

    run_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/02-29_11:04___COCO_HigherHRNet/02-29_11:04"

    ckpt_path = f"{run_path}/checkpoints/best.pt"

    model = load_model(cfg_path, ckpt_path)

    root = str(DS_ROOT / f"{model.ds_name}/raw")
    split = "val2017"

    ds = BaseImageDataset(root=root, split=split, transform=None)
    results = evaluate_dataset(ds, model)

    results_path = f"{run_path}/{split}_results.json"
    save_json(results, results_path)


if __name__ == "__main__":
    main()
