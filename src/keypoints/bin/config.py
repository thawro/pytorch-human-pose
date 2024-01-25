from src.keypoints.config import (
    TransformConfig,
    DatasetConfig,
    DataloaderConfig,
    SetupConfig,
    TrainerConfig,
    Config,
    _dataset_name,
    _mode,
    _architectures,
)

EXPERIMENT_NAME = "test"

EXP_MODE = "test"
# EXP_MODE = "debug"

NAME_PREFIX = ""
BATCH_SIZE = 36


if EXP_MODE == "debug":
    LIMIT_BATCHES = 10
    LOG_EVERY_N_STEPS = -1
else:
    LIMIT_BATCHES = -1
    LOG_EVERY_N_STEPS = -5

if LIMIT_BATCHES != -1:
    EXPERIMENT_NAME = "debug"


def create_config(
    dataset_name: _dataset_name,
    mode: _mode,
    arch: _architectures,
    device_id: int,
    ckpt_path: str | None = None,
    distributed: bool = True,
    is_train: bool = True,
) -> Config:
    dataset_cfg = DatasetConfig(name=dataset_name, mode=mode, out_size=[512, 512])

    transform_cfg = TransformConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        out_size=dataset_cfg.out_size,
        symmetric_keypoints=dataset_cfg.symmetric_keypoints,
    )

    dataloader_cfg = DataloaderConfig(batch_size=BATCH_SIZE, transform=transform_cfg)

    trainer_cfg = TrainerConfig(
        device_id=device_id,
        max_epochs=300,
        limit_batches=LIMIT_BATCHES,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )

    setup_cfg = SetupConfig(
        distributed=distributed,
        experiment_name=EXPERIMENT_NAME,
        name_prefix=NAME_PREFIX,
        seed=42,
        dataset=dataset_cfg.name,
        ckpt_path=ckpt_path,
        mode=dataset_cfg.mode,
        arch=arch,
        is_train=is_train,
    )

    return Config(
        setup=setup_cfg,
        dataloader=dataloader_cfg,
        dataset=dataset_cfg,
        trainer=trainer_cfg,
    )
