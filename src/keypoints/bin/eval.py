import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.keypoints.bin.inference import InferenceKeypointsModel, _set_paths, load_model
from src.logger.pylogger import log
from src.utils.config import DS_ROOT, YAML_EXP_PATH
from src.utils.files import load_yaml, save_json
from src.utils.model import seed_everything


def evaluate_dataset(dataset: BaseImageDataset, model: InferenceKeypointsModel):
    filepaths = dataset.images_filepaths.tolist()
    n_examples = len(dataset)
    # n_examples = 100
    results = []
    with torch.no_grad():
        for idx in tqdm(range(n_examples)):
            image_path = filepaths[idx]  # .decode("utf-8")
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
            kpts_coords = result.kpts_coords

            obj_scores = result.obj_scores

            num_obj = len(obj_scores)

            for i in range(num_obj):
                kpts = kpts_coords[i]
                scores = obj_scores[i]
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


def eval_coco(annots_path: str, results_path: str):
    # iouType: one of ["segm", "bbox", "keypoints"]

    cocoGt = COCO(annots_path)

    cocoDt = cocoGt.loadRes(results_path)
    img_ids = list(set([ann["image_id"] for i, ann in cocoDt.anns.items()]))
    log.info(f"Evaluating {len(img_ids)} samples")

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="keypoints")
    cocoEval.params.imgIds = img_ids
    cocoEval.params.useSegm = None

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main() -> None:
    seed_everything(42)
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")

    run_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-02_08:35__COCO_HigherHRNet/03-02_08:35"
    run_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35"
    ckpt_path = f"{run_path}/checkpoints/best.pt"

    model = load_model(cfg_path, ckpt_path, device_id=1)

    root = str(DS_ROOT / "COCO/raw")
    split = "val2017"

    BaseImageDataset._set_paths = _set_paths
    ds = BaseImageDataset(root=root, split=split, transform=None)
    ds._set_paths()

    results = evaluate_dataset(ds, model)

    results_path = f"{run_path}/{split}_results.json"
    save_json(results, results_path)

    gt_annot_path = str(DS_ROOT / "COCO/raw/annotations/person_keypoints_val2017.json")
    eval_coco(gt_annot_path, results_path)


if __name__ == "__main__":
    main()
