from src.keypoints.config import KeypointsConfig
from src.keypoints.datasets.coco import CocoKeypointsDataset
from src.logger.pylogger import log
from src.utils.config import YAML_EXP_PATH

if __name__ == "__main__":
    log.info("Saving COCO keypoints annotations and crowd masks to files")
    cfg_path = YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml"
    cfg_dict = KeypointsConfig.from_yaml_to_dict(cfg_path)

    log.info("Started train split saving")
    train_ds = CocoKeypointsDataset(**cfg_dict["dataloader"]["train_ds"], transform=None)
    train_ds._save_annots_to_files()
    log.info("Ended train split saving")

    log.info("Started val split saving")
    val_ds = CocoKeypointsDataset(**cfg_dict["dataloader"]["val_ds"], transform=None)
    val_ds._save_annots_to_files()
    log.info("Ended val split saving")
