import argparse
from typing import Literal

from src.base.bin.inference import prepare_inference_config
from src.base.datasets import DirectoryDataset
from src.classification.config import ClassificationConfig
from src.classification.datasets import ImagenetClassificationDataset
from src.logger.pylogger import log, log_breaking_point
from src.utils.config import YAML_EXP_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ClassificationHRNet Inference",
        description="Perform inference of ClassificationHRNet neural network trained on ImageNet dataset",
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=str,
        help="Inference mode. If 'val', then ImageNet val split is used, if 'custom', then dirpath is used",
        default="val",
    )
    parser.add_argument(
        "--dirpath",
        action="store",
        type=str,
        help="Path to directory with images for inference. Used only when 'mode' is set to 'custom'",
        default=None,
    )
    args, unknown = parser.parse_known_args()
    log.info(f"Parsed args: \n{args}")
    return args


def main() -> None:
    log_breaking_point("Starting inference", n_top=1, n_bottom=1, top_char="*", bottom_char="*")
    cfg_path = str(YAML_EXP_PATH / "classification" / "hrnet_32.yaml")
    cfg: ClassificationConfig = prepare_inference_config(cfg_path, ClassificationConfig)

    ds_cfg = cfg.dataloader.val_ds
    ds = ImagenetClassificationDataset(root=ds_cfg.root, split=ds_cfg.split)
    idx2label = ds.idx2label

    model = cfg.create_inference_model(idx2label=idx2label, device="cuda:0")

    args = parse_args()

    if args.mode == "custom":
        ds = DirectoryDataset(args.dirpath)

    ds.perform_inference(model=model, idx=0, load_annot=False)


if __name__ == "__main__":
    main()
