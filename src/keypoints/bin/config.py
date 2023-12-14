from src.keypoints.config import (
    TransformConfig,
    DataloaderConfig,
    SetupConfig,
    OptimizerConfig,
    Config,
)

BATCH_SIZE = 128
EXPERIMENT_NAME = "test"
# DATASET = "COCO"
DATASET = "MPII"

LIMIT_BATCHES = -1
LOG_EVERY_N_STEPS = -1

CKPT_PATH = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/12-12-2023_20:07:56_SPPE_MPII_LR(0.001)_HRNet/checkpoints/last.pt"
# CKPT_PATH = None

MODE = "SPPE"
ARCHITECTURE = "HRNet"
LR = 1e-3
OUT_SIZE = (256, 256)

if LIMIT_BATCHES != -1:
    EXPERIMENT_NAME = "debug"

transform_cfg = TransformConfig(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], out_size=OUT_SIZE
)

dataloader_cfg = DataloaderConfig(batch_size=BATCH_SIZE, transform=transform_cfg)

setup_cfg = SetupConfig(
    experiment_name=EXPERIMENT_NAME,
    seed=42,
    device="cuda",
    dataset=DATASET,
    max_epochs=300,
    limit_batches=LIMIT_BATCHES,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    ckpt_path=CKPT_PATH,
    mode=MODE,
    arch=ARCHITECTURE,
)

optimizer_cfg = OptimizerConfig(lr=LR)

cfg = Config(setup_cfg, dataloader_cfg, optimizer_cfg)
