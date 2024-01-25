import torch
from tqdm.auto import tqdm

from src.utils.config import DS_ROOT
from src.utils.model import seed_everything
from src.keypoints.bin.inference import load_model, MPPEInferenceKeypointsModel
from src.base.datasets import BaseImageDataset

import numpy as np
from PIL import Image
from src.utils.files import load_yaml, save_json
import cv2
import os
from pathlib import Path


def evaluate_dataset(dataset: BaseImageDataset, model: MPPEInferenceKeypointsModel):
    n_examples = len(dataset)
    # n_examples = 100
    results = []
    try:
        with torch.no_grad():
            for idx in tqdm(range(n_examples)):
                image_path = dataset.images_filepaths[idx]
                annot_path = image_path.replace(".jpg", ".yaml").replace(
                    "images/", "annots/"
                )
                image = np.asarray(Image.open(image_path))
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                if os.path.isfile(annot_path):
                    annot = load_yaml(annot_path)
                    image_id = int(
                        annot["filename"]
                        .replace(".jpg", "")
                        .replace("_valid", "")
                        .lstrip("0")
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
    except Exception as e:
        save_json(results, "results.json")
        raise e
    save_json(results, "results.json")


def main() -> None:
    dataset = "COCO"
    model = load_model(dataset)

    root = str(DS_ROOT / f"{dataset}/raw")
    split = "val2017"

    # root = str(DS_ROOT / f"{dataset}/HumanPose")
    # split = "val"
    ds = BaseImageDataset(root=root, split=split, transform=None)

    seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    evaluate_dataset(ds, model)


if __name__ == "__main__":
    main()
