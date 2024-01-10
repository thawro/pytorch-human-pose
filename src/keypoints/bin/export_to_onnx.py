from src.keypoints.bin.utils import create_model
from src.keypoints.bin.config import cfg
import torch

RUN_PATH = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/13-12-2023_17:26:01_SPPE_MPII_LR(0.001)_HRNet"
ckpt_name = "last"
cfg.setup.ckpt_path = f"{RUN_PATH}/checkpoints/{ckpt_name}.pt"
cfg.setup.mode = "SPPE"
cfg.setup.arch = "HRNet"
cfg.trainer.device = "cuda:0"


def main():
    model = create_model(cfg)
    ckpt = torch.load(cfg.setup.ckpt_path, map_location=cfg.setup.device)
    state_dict = ckpt["module"]["model"]
    model.load_state_dict(state_dict)
    model.eval()

    model.export_to_onnx(f"{RUN_PATH}/model/onnx/{ckpt_name}.onnx")


if __name__ == "__main__":
    main()
