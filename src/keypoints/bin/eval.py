from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm

from src.base.bin.eval import prepare_eval_config
from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.keypoints.model import InferenceKeypointsModel
from src.logger.pylogger import log, log_breaking_point
from src.utils.config import DS_ROOT, NOW
from src.utils.files import read_text_file, save_json, save_yaml


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


def eval_coco(annots_path: str, results_path: str) -> str:
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
    return cocoEval.stats


def main() -> None:
    log_breaking_point("Starting Evaluation", n_top=1, n_bottom=1, top_char="*", bottom_char="*")
    eval_split = "val2017"

    gt_annot_path = str(DS_ROOT / f"COCO/annotations/person_keypoints_{eval_split}.json")

    run_path = "results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35"
    cfg_path = f"{run_path}/config.yaml"
    ckpt_path = f"{run_path}/checkpoints/best.pt"

    eval_results_dir = f"{run_path}/evaluation_results/{NOW}"
    Path(eval_results_dir).mkdir(exist_ok=True, parents=True)
    log.info(f"Evaluation results will be saved in {eval_results_dir} directory")

    results_filepath = f"{eval_results_dir}/{eval_split}_results.json"
    eval_config_filepath = f"{eval_results_dir}/config.yaml"
    coco_str_eval_filepath = f"{eval_results_dir}/coco_output.txt"

    cfg: KeypointsConfig = prepare_eval_config(cfg_path, ckpt_path, KeypointsConfig)
    model = cfg.create_inference_model(device="cuda:0")

    if "val" in eval_split:
        ds_cfg = cfg.dataloader.val_ds
    elif "train" in eval_split:
        ds_cfg = cfg.dataloader.train_ds
    else:
        raise ValueError("Only val2017 and train2017 splits are available for evaluation")

    ds = CocoKeypointsDataset(root=ds_cfg.root, split=ds_cfg.split)

    results = evaluate_dataset(model, ds)

    save_json(results, results_filepath)
    save_yaml(cfg.to_dict(), eval_config_filepath)

    log.info(f"Saved results in '{results_filepath}'")
    log.info(f"Saved config in '{eval_config_filepath}'")

    with open(coco_str_eval_filepath, "w") as f:
        with redirect_stdout(f):
            eval_coco(gt_annot_path, results_filepath)

    log.info(f"Saved coco output in '{coco_str_eval_filepath}'")
    coco_output = read_text_file(coco_str_eval_filepath)
    coco_output = "\n".join(coco_output)
    log.info(f"COCO output:\n{coco_output}")


if __name__ == "__main__":
    main()
