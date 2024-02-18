from dataclasses import dataclass, fields
from src.utils import NOW, RESULTS_PATH
from pathlib import Path
from src.utils.files import load_yaml
from dacite import from_dict
from abc import abstractmethod
from .datamodule import DataModule
from .module import BaseModule
from .trainer import Trainer
from .callbacks import (
    BaseCallback,
    ModelSummary,
    MetricsLogger,
    SaveModelCheckpoint,
    MetricsSaverCallback,
    MetricsPlotterCallback,
)
import os

from src.logger.loggers import TerminalLogger

from src.logger.pylogger import log, get_file_pylogger


@dataclass
class AbstractConfig:
    def to_dict(self) -> dict:
        dct = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if hasattr(field_value, "to_dict"):
                dct[field_name] = field_value.to_dict()
            else:
                dct[field_name] = field_value
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        return from_dict(data_class=cls, data=dct)


@dataclass
class TransformConfig(AbstractConfig):
    mean: tuple[float, ...] | list[float]
    std: tuple[float, ...] | list[float]
    out_size: list[int] | tuple[int, int]


@dataclass
class DatasetConfig(AbstractConfig):
    name: str
    transform: TransformConfig


@dataclass
class DataloaderConfig(AbstractConfig):
    batch_size: int
    pin_memory: bool
    num_workers: int
    dataset: DatasetConfig


@dataclass
class ModelConfig(AbstractConfig):
    architecture: str


@dataclass
class TrainerConfig(AbstractConfig):
    accelerator: str
    max_epochs: int
    limit_batches: int
    log_every_n_steps: int
    use_distributed: bool
    use_fp16: bool


@dataclass
class SetupConfig(AbstractConfig):
    seed: int
    experiment_name: str
    name_prefix: str
    is_train: bool
    ckpt_path: str | None
    pretrained_ckpt_path: str | None


@dataclass
class BaseConfig(AbstractConfig):
    setup: SetupConfig
    dataloader: DataloaderConfig
    model: ModelConfig
    trainer: TrainerConfig
    
    def __post_init__(self):        
        self.file_log = get_file_pylogger(f"{self.logs_path}/{self.device}_log.log", "log_file")
        log.handlers.append(self.file_log.handlers[0])

    def log_info(self, msg: str) -> None:
        log.info(f"[{self.device}] {msg}")

    @property
    def device(self) -> str:
        if self.trainer.accelerator == "gpu":
            if self.trainer.use_distributed and "LOCAL_RANK" in os.environ:
                device_id = int(os.environ["LOCAL_RANK"])
            else:
                device_id = 0
            device = f"cuda:{device_id}"
        else:
            device = "cpu"
        return device

    @property
    def run_name(self) -> str:
        dataset = f"_{self.dataloader.dataset.name}"
        name = f"_{self.setup.name_prefix}"
        architecture = f"_{self.model.architecture}"
        return f"{NOW}_{name}{dataset}{architecture}"

    @property
    def logs_path(self) -> str:
        ckpt_path = self.setup.ckpt_path
        is_train = self.setup.is_train
        if ckpt_path is None:
            exp_name = "debug" if self.is_debug else self.setup.experiment_name
            _logs_path = str(RESULTS_PATH / exp_name / self.run_name / NOW)
        else:
            ckpt_path = Path(ckpt_path)
            loaded_ckpt_run_path = ckpt_path.parent.parent
            loaded_run_path = loaded_ckpt_run_path.parent
            if is_train:
                _logs_path = str(loaded_run_path / NOW)
            else:
                _logs_path = str(loaded_ckpt_run_path)
        Path(_logs_path).mkdir(exist_ok=True, parents=True)
        return _logs_path

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    @classmethod
    def from_yaml(cls, filepath: str | Path):
        cfg = load_yaml(filepath)
        return cls.from_dict(cfg)

    @abstractmethod
    def create_datamodule(self) -> DataModule:
        raise NotImplementedError()

    @abstractmethod
    def create_module(self) -> BaseModule:
        raise NotImplementedError()

    def create_callbacks(self) -> list[BaseCallback]:
        self.log_info("..Creating Callbacks..")
        callbacks = [
            MetricsPlotterCallback(),
            MetricsSaverCallback(),
            MetricsLogger(),
            ModelSummary(depth=4),
            SaveModelCheckpoint(
                name="best", metric="loss", last=True, mode="min", stage="val"
            ),
        ]
        return callbacks

    @abstractmethod
    def create_trainer(self) -> Trainer:
        self.log_info("..Creating Trainer..")
        logger = TerminalLogger(self.logs_path, config=self.to_dict())
        callbacks = self.create_callbacks()
        return Trainer(logger=logger, callbacks=callbacks, file_log=self.file_log, **self.trainer.to_dict())


if __name__ == "__main__":
    cfg = BaseConfig.from_yaml(
        "/home/thawro/Desktop/projects/pytorch-human-pose/experiments/classification/hrnet_32.yaml"
    )
    print(cfg)
