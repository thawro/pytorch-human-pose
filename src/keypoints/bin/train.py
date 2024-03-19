"""Train the keypoints estimation model"""

from src.base.bin.train import train
from src.keypoints.config import KeypointsConfig
from src.utils.config import YAML_EXP_PATH


def main() -> None:
    cfg_path = YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml"
    cfg = KeypointsConfig.from_yaml_to_dict(cfg_path)
    train(cfg, KeypointsConfig)


if __name__ == "__main__":
    main()
