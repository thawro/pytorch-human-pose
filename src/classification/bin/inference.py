from src.base.bin.inference import prepare_inference_config
from src.classification.config import ClassificationConfig
from src.classification.datasets import ImagenetClassificationDataset
from src.logger.pylogger import log_breaking_point
from src.utils.config import YAML_EXP_PATH


def main() -> None:
    log_breaking_point("Starting inference", n_top=1, n_bottom=1, top_char="*", bottom_char="*")
    cfg_path = str(YAML_EXP_PATH / "classification" / "hrnet_32.yaml")
    cfg: ClassificationConfig = prepare_inference_config(cfg_path, ClassificationConfig)

    ds_cfg = cfg.dataloader.val_ds
    ds = ImagenetClassificationDataset(root=ds_cfg.root, split=ds_cfg.split)

    model = cfg.create_inference_model(idx2label=ds.idx2label, device="cuda:0")
    ds.perform_inference(model=model, idx=0, load_annot=False)


if __name__ == "__main__":
    main()
