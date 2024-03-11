"""Train the keypoints estimation model"""

from src.base.bin.train import train
from src.keypoints.config import KeypointsConfig
from src.utils.config import YAML_EXP_PATH
from src.utils.files import load_yaml


def main() -> None:
    cfg_path = YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml"
    cfg = load_yaml(cfg_path)

    pretrained_ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-15_10:12___imagenet_HRNet/02-19_09:14/checkpoints/best.pt"

    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35/checkpoints/best.pt"
    ckpt_path = None

    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg["setup"]["pretrained_ckpt_path"] = pretrained_ckpt_path
    cfg["trainer"]["limit_batches"] = -1
    cfg["trainer"]["use_DDP"] = True

    cfg = KeypointsConfig.from_dict(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
