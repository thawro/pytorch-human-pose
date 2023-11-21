from src.keypoints.config import (
    TransformConfig,
    DataloaderConfig,
    SetupConfig,
    OptimizerConfig,
    Config,
)

BATCH_SIZE = 24
EXPERIMENT_NAME = "test"
# DATASET = "COCO"
DATASET = "MPII"

LIMIT_BATCHES = -1
LOG_EVERY_N_STEPS = -1

CKPT_PATH = "/home/shate/Desktop/projects/pytorch-human-pose/results/test/17-11-2023_19:00:22_train_MPII_LR(0.001)/checkpoints/last.pt"
CKPT_PATH = None

MODE = "train"
LR = 1e-3
SIZE = 256

if LIMIT_BATCHES != -1:
    EXPERIMENT_NAME = "debug"

transform_cfg = TransformConfig(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=SIZE
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
    multiobj=False,
)

optimizer_cfg = OptimizerConfig(lr=LR)

cfg = Config(setup_cfg, dataloader_cfg, optimizer_cfg)
