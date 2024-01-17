import torch
from tqdm.auto import tqdm

from src.utils.config import DS_ROOT
from src.utils.model import seed_everything
from src.keypoints.bin.inference import load_model
from src.base.datasets import BaseImageDataset

import numpy as np
from PIL import Image
from src.utils.files import load_yaml, save_json
import cv2


def main() -> None:
    dataset = "COCO"
    model = load_model(dataset)
    ds = BaseImageDataset(
        root=str(DS_ROOT / f"{dataset}/HumanPose"),
        split="val",
        transform=None,
    )

    seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    n_examples = len(ds)
    n_examples = 100
    results = []
    with torch.no_grad():
        for idx in tqdm(range(n_examples)):
            image_path = ds.images_filepaths[idx + 1]
            annot_path = image_path.replace(".jpg", ".yaml").replace(
                "images/", "annots/"
            )
            image = np.asarray(Image.open(image_path))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            annot = load_yaml(annot_path)
            image_id = int(
                annot["filename"].replace(".jpg", "").replace("_valid", "").lstrip("0")
            )

            result = model(image, None)
            # final_plot, raw_image = result.plot()
            # cv2.imshow("KPTS", raw_image)

            # cv2.waitKey()
            pred_kpts = result.pred_keypoints

            pred_scores = result.pred_scores

            num_obj = len(pred_scores)

            for i in range(num_obj):
                kpts = pred_kpts[i]
                scores = pred_scores[i]
                num_kpts = len(kpts)

                coco_kpts = np.zeros((num_kpts * 3,), dtype=np.int32)
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
    save_json(results, "results.json")


if __name__ == "__main__":
    main()
