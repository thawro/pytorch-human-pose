from src.base.bin.train import train
from src.base.callbacks import (
    BaseCallback,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    MetricsLogger,
    ModelSummary,
    SaveModelCheckpoint,
)

from src.utils.config import RESULTS_PATH, YAML_EXP_PATH
from src.utils.files import load_yaml

from src.classification.config import ClassificationConfig

# top-5 validation error of 6.5%
# top-1 validation error of 22.7%


def create_callbacks() -> list[BaseCallback]:
    callbacks = [
        MetricsPlotterCallback(),
        MetricsSaverCallback(),
        MetricsLogger(),
        ModelSummary(depth=4),
        SaveModelCheckpoint(
            name="best",
            metric="loss",
            last=True,
            mode="min",
            stage="val",
        ),
    ]
    return callbacks


def main():
    cfg_path = YAML_EXP_PATH / "classification" / "hrnet_32.yaml"
    cfg = load_yaml(cfg_path)

    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-09_23:08___ImageNet_HRNet/02-09_23:08/checkpoints/last.pt"
    # ckpt_path = None

    cfg["setup"]["ckpt_path"] = ckpt_path
    # cfg["trainer"]["limit_batches"] = 5
    cfg["trainer"]["use_distributed"] = True

    cfg = ClassificationConfig.from_dict(cfg)

    datamodule = cfg.create_datamodule()
    module = cfg.create_module()

    callbacks = create_callbacks()

    trainer = cfg.create_trainer(callbacks=callbacks)

    train(
        trainer,
        module,
        datamodule,
        pretrained_ckpt_path=None,
        ckpt_path=ckpt_path,
        seed=42,
    )


if __name__ == "__main__":
    main()
