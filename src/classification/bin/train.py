from src.base.bin.train import train
from src.classification.config import ClassificationConfig
from src.utils.config import YAML_EXP_PATH

# top-5 validation error of 6.5%
# top-1 validation error of 22.7%


def main():
    cfg_path = YAML_EXP_PATH / "classification" / "hrnet_32.yaml"
    cfg = ClassificationConfig.from_yaml_to_dict(cfg_path)
    train(cfg, ClassificationConfig)


if __name__ == "__main__":
    main()
