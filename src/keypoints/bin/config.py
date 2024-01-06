from src.keypoints.config import (
    TransformConfig,
    DataloaderConfig,
    SetupConfig,
    OptimizerConfig,
    Config,
)
from src.utils.config import RESULTS_PATH


EXPERIMENT_NAME = "test"

# DATASET = "COCO"
DATASET = "MPII"

MODE = "SPPE"
MODE = "MPPE"

LIMIT_BATCHES = -1
LOG_EVERY_N_STEPS = -7

CKPT_PATH = f"{str(RESULTS_PATH)}/{EXPERIMENT_NAME}/01-06_16:21__sigmoid_MPPE_MPII/01-06_16:21/checkpoints/last.pt"
CKPT_PATH = None

NAME_PREFIX = "sigmoid"

# ARCHITECTURE = "HRNet"
ARCHITECTURE = "HigherHRNet"

LR = 1e-3

if MODE == "SPPE":
    OUT_SIZE = (256, 256)
    BATCH_SIZE = 184
else:
    OUT_SIZE = (512, 512)
    BATCH_SIZE = 36


if LIMIT_BATCHES != -1:
    EXPERIMENT_NAME = "debug"

transform_cfg = TransformConfig(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], out_size=list(OUT_SIZE)
)

dataloader_cfg = DataloaderConfig(batch_size=BATCH_SIZE, transform=transform_cfg)

setup_cfg = SetupConfig(
    experiment_name=EXPERIMENT_NAME,
    name_prefix=NAME_PREFIX,
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
