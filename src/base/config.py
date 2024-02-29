import logging
from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

from dacite import from_dict

from src.logger.loggers import Loggers, MLFlowLogger, Status
from src.logger.pylogger import get_file_pylogger, log
from src.utils import NOW, RESULTS_PATH
from src.utils.config import LOG_DEVICE_ID
from src.utils.files import load_yaml
from src.utils.utils import get_device_and_id

from .callbacks import (
    BaseCallback,
    LogsLoggerCallback,
    MetricsLogger,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
)
from .datamodule import DataModule
from .module import BaseModule
from .trainer import Trainer


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
        self.device, self.device_id = get_device_and_id(
            self.trainer.accelerator, self.trainer.use_distributed
        )
        if not self.setup.is_train:
            return
        logs_path = Path(self.log_path) / "logs"
        logs_path.mkdir(exist_ok=True, parents=True)
        file_log = get_file_pylogger(f"{logs_path}/{self.device}_log.log", "log_file")
        # insert handler to enable file logs in command line aswell
        log.handlers.insert(0, file_log.handlers[0])
        for handler in log.handlers:
            formatter = handler.formatter
            if formatter is not None and hasattr(formatter, "_set_device"):
                handler.formatter._set_device(self.device, self.device_id)
        self.logger = self.create_logger(file_log)
        if self.device_id == LOG_DEVICE_ID:
            self.logger.start_run()

    @property
    def run_name(self) -> str:
        ckpt_path = self.setup.ckpt_path
        if ckpt_path is None:
            dataset = f"_{self.dataloader.dataset.name}"
            name = f"_{self.setup.name_prefix}"
            architecture = f"_{self.model.architecture}"
            return f"{NOW}_{name}{dataset}{architecture}"
        else:
            # ckpt_path is like:
            # "<proj_root>/results/<exp_name>/<run_name>/<timestamp>/checkpoints/<ckpt_name>.pt"
            # so run_name is -4 idx after split
            run_name = ckpt_path.split("/")[-4]
            return run_name

    @property
    def log_path(self) -> str:
        ckpt_path = self.setup.ckpt_path
        is_train = self.setup.is_train
        if ckpt_path is None:
            exp_name = "debug" if self.is_debug else self.setup.experiment_name
            _log_path = str(RESULTS_PATH / exp_name / self.run_name / NOW)
        else:
            ckpt_path = Path(ckpt_path)
            loaded_ckpt_run_path = ckpt_path.parent.parent
            loaded_run_path = loaded_ckpt_run_path.parent
            if is_train:
                _log_path = str(loaded_run_path / NOW)
            else:
                _log_path = str(loaded_ckpt_run_path)
        Path(_log_path).mkdir(exist_ok=True, parents=True)
        return _log_path

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
        log.info("..Creating Callbacks..")
        callbacks = [
            MetricsPlotterCallback(),
            MetricsSaverCallback(),
            MetricsLogger(),
            LogsLoggerCallback(),
            ModelSummary(depth=5),
            SaveModelCheckpoint(name="best", metric="loss", last=True, mode="min", stage="val"),
        ]
        return callbacks

    def create_logger(self, file_log: logging.Logger) -> Loggers:
        log.info("..Creating Logger..")
        loggers = [
            # TerminalLogger(self.logs_path, config=self.to_dict()),
            MLFlowLogger(
                self.log_path,
                config=self.to_dict(),
                experiment_name=self.setup.experiment_name,
                run_name=self.run_name,
            )
        ]
        logger = Loggers(loggers, self.device_id, file_log)
        return logger

    def create_trainer(self) -> Trainer:
        log.info("..Creating Trainer..")
        try:
            callbacks = self.create_callbacks()
            trainer = Trainer(
                logger=self.logger,
                callbacks=callbacks,
                **self.trainer.to_dict(),
            )
            return trainer
        except Exception as e:
            log.error(str(e))
            self.logger.finalize(Status.FAILED)
            raise e


if __name__ == "__main__":
    from src.utils.config import YAML_EXP_PATH

    cfg = BaseConfig.from_yaml(str(YAML_EXP_PATH / "classification/hrnet_32.yaml"))
    print(cfg)
