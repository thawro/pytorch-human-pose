"""Train the keypoints estimation model"""

from src.base.bin.train import train
from src.keypoints.config import KeypointsConfig
from src.utils.config import YAML_EXP_PATH


def main() -> None:
    cfg_path = YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml"
    cfg = KeypointsConfig.from_yaml_to_dict(cfg_path)

    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/keypoints/03-05_15:47__COCO_HigherHRNet/03-08_07:35/checkpoints/best.pt"
    ckpt_path = None

    cfg["setup"]["ckpt_path"] = ckpt_path

    train(cfg, KeypointsConfig)


if __name__ == "__main__":
    main()
