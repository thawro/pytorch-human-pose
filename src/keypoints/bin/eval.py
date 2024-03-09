from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm

from src.keypoints.bin.inference import (
    InferenceKeypointsModel,
    load_model,
    prepare_inference_config,
)
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.utils.config import DS_ROOT, YAML_EXP_PATH
from src.utils.files import save_json
from src.utils.model import seed_everything


def evaluate_dataset(model: InferenceKeypointsModel, dataset: CocoKeypointsDataset):
    results = []
    num_samples = len(dataset)
    for idx in tqdm(range(num_samples)):
        image_filepath = dataset.images_filepaths[idx]
        image_id = int(Path(image_filepath).stem.lstrip("0"))

        raw_image = dataset.load_image(idx)

        result = model(raw_image, annot=None)
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
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="keypoints")
    cocoEval.params.imgIds = img_ids
    cocoEval.params.useSegm = None

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main() -> None:
    seed_everything(42)
    cfg_path = str(YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml")

    run_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35"
    ckpt_path = f"{run_path}/checkpoints/best.pt"

    coco_eval_results_dir = f"{run_path}/coco_eval_results"

    cfg = prepare_inference_config(cfg_path, ckpt_path)

    ds_cfg = cfg.dataloader.val_ds
    ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)

    model = load_model(cfg)
    results = evaluate_dataset(model, ds)

    filename = (
        f"{ds_cfg.split}_results"
        f"_tagThr({cfg.inference.tag_thr})"
        f"_detThr({cfg.inference.det_thr})"
        f"_useFlip({cfg.inference.use_flip})"
        f"_inputSize({cfg.inference.input_size})"
    )
    results_path = f"{coco_eval_results_dir}/{filename}.json"
    save_json(results, results_path)

    gt_annot_path = str(DS_ROOT / "COCO/raw/annotations/person_keypoints_val2017.json")
    eval_coco(gt_annot_path, results_path)


if __name__ == "__main__":
    main()
