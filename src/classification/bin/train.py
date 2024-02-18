from src.base.bin.train import train


from src.utils.config import YAML_EXP_PATH
from src.utils.files import load_yaml

from src.classification.config import ClassificationConfig

# top-5 validation error of 6.5%
# top-1 validation error of 22.7%


def main():
    cfg_path = YAML_EXP_PATH / "classification" / "hrnet_32.yaml"
    # cfg_path = YAML_EXP_PATH / "classification" / "original_hrnet_32.yaml"

    cfg = load_yaml(cfg_path)

    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-15_10:12___imagenet_HRNet/02-18_11:59/checkpoints/last.pt"
    ckpt_path = None

    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg["setup"]["pretrained_ckpt_path"] = None
    cfg["trainer"]["limit_batches"] = 5
    cfg["trainer"]["use_distributed"] = True

    
    cfg = ClassificationConfig.from_dict(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
