import logging
from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Type

from dacite import from_dict
from torch import nn

from src.logger.loggers import Loggers, MLFlowLogger, Status
from src.logger.pylogger import get_file_pylogger, log
from src.utils import NOW, RESULTS_PATH
from src.utils.config import LOG_DEVICE_ID
from src.utils.files import load_yaml
from src.utils.utils import get_device_and_id

from .callbacks import (
    ArtifactsLoggerCallback,
    BaseCallback,
    DatasetExamplesCallback,
    MetricsLogger,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
)
from .datamodule import DataModule
from .model import BaseInferenceModel
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
                if isinstance(field_value, dict):  # Config class nested in dict (e.g. optimizers)
                    init_dict = {}
                    for k, v in field_value.items():
                        if hasattr(v, "to_dict"):
                            init_dict[k] = v.to_dict()
                        else:
                            init_dict[k] = v
                    dct[field_name] = init_dict
                else:
                    dct[field_name] = field_value
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        return from_dict(data_class=cls, data=dct)


@dataclass
class TransformConfig(AbstractConfig):
    out_size: int | tuple[int, int] | list[int]
    mean: list[float]
    std: list[float]


@dataclass
class DatasetConfig(AbstractConfig):
    root: str
    split: str


@dataclass
class DataloaderConfig(AbstractConfig):
    batch_size: int
    pin_memory: bool
    num_workers: int
    train_ds: DatasetConfig
    val_ds: DatasetConfig


@dataclass
class NetConfig(AbstractConfig):
    params: dict


@dataclass
class TrainerConfig(AbstractConfig):
    accelerator: str
    max_epochs: int
    limit_batches: int
    use_DDP: bool
    sync_batchnorm: bool
    use_compile: bool


@dataclass
class SetupConfig(AbstractConfig):
    seed: int
    experiment_name: str
    is_train: bool
    ckpt_path: str | None
    pretrained_ckpt_path: str | None
    deterministic: bool
    architecture: str
    dataset: str
    run_name: str | None = None

    def _auto_run_name(self) -> str:
        if self.ckpt_path is None:
            # new run
            return f"{self.timestamp}__{self.dataset}_{self.architecture}"
        # resumed run
        # ckpt_path is like:
        # "<proj_root>/results/<exp_name>/<run_name>/<timestamp>/checkpoints/<ckpt_name>.pt"
        # so run_name is -4 idx after split
        run_name = self.ckpt_path.split("/")[-4]
        return run_name

    def __post_init__(self):
        self.timestamp = NOW
        if self.run_name is None:
            self.run_name = self._auto_run_name()


@dataclass
class CUDNNConfig(AbstractConfig):
    benchmark: bool
    deterministic: bool
    enabled: bool


@dataclass
class OptimizerConfig(AbstractConfig):
    name: str
    params: dict


@dataclass
class LRSchedulerConfig(AbstractConfig):
    name: str
    interval: Literal["epoch", "step"]
    params: dict


@dataclass
class ModuleConfig(AbstractConfig):
    optimizers: dict[str, OptimizerConfig]
    lr_schedulers: dict[str, LRSchedulerConfig]


@dataclass
class InferenceConfig(AbstractConfig):
    use_flip: bool


@dataclass
class BaseConfig(AbstractConfig):
    setup: SetupConfig
    cudnn: CUDNNConfig
    dataloader: DataloaderConfig
    transform: TransformConfig
    net: NetConfig
    module: ModuleConfig
    trainer: TrainerConfig
    inference: InferenceConfig

    def __post_init__(self):
        if self.is_debug:
            self.setup.experiment_name = "debug"
        self.device, self.device_id = get_device_and_id(
            self.trainer.accelerator, self.trainer.use_DDP
        )
        if not self.setup.is_train:  # no logging for eval/test/inference
            return
        self._initialize_logger()
        if self.device_id == LOG_DEVICE_ID:
            self.logger.start_run()

    def _initialize_logger(self):
        file_log_dirpath = f"{self.log_path}/logs"
        Path(file_log_dirpath).mkdir(exist_ok=True, parents=True)
        file_log_filepath = f"{file_log_dirpath}/{self.device}_log.log"
        file_log = get_file_pylogger(file_log_filepath, "log_file")
        # insert handler to enable file logs in command line aswell
        log.handlers.insert(0, file_log.handlers[0])
        log.info(f"..Saving {self.device} logs to {file_log_filepath}..")
        for handler in log.handlers:
            formatter = handler.formatter
            if formatter is not None and hasattr(formatter, "_set_device"):
                handler.formatter._set_device(self.device, self.device_id)
        self.logger = self.create_logger(file_log)

    @property
    def log_path(self) -> str:
        ckpt_path = self.setup.ckpt_path
        if ckpt_path is None or self.is_debug:
            _log_path = str(RESULTS_PATH / self.setup.experiment_name / self.setup.run_name / NOW)
        else:
            loaded_run_path = Path(ckpt_path).parent.parent.parent
            _log_path = str(loaded_run_path / NOW)
        Path(_log_path).mkdir(exist_ok=True, parents=True)
        return _log_path

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def get_optimizers_params(self) -> dict[str, dict]:
        log.info("..Parsing Optimizers config..")
        return {
            name: optimizer_cfg.to_dict() for name, optimizer_cfg in self.module.optimizers.items()
        }

    def get_lr_schedulers_params(self) -> dict[str, dict]:
        log.info("..Parsing LR Schedulers config..")
        return {
            name: scheduler_cfg.to_dict()
            for name, scheduler_cfg in self.module.lr_schedulers.items()
        }

    @classmethod
    def from_yaml(cls, filepath: str | Path):
        cfg = load_yaml(filepath)
        return cls.from_dict(cfg)

    @property
    @abstractmethod
    def architectures(self) -> dict[str, Type[nn.Module]]:
        raise NotImplementedError()

    def create_net(self) -> nn.Module:
        arch = self.setup.architecture
        params_repr = "\n".join([f"     {k}: {v}" for k, v in self.net.params.items()])
        arch_repr = f"{arch}\n{params_repr}"
        log.info(f"..Creating {arch} Neural Network..\n{arch_repr}")

        assert (
            arch in self.architectures
        ), f"model.architecture must be one of {list(self.architectures.keys())}"

        ArchitectureClass = self.architectures[arch]
        net = ArchitectureClass(**self.net.params)
        return net

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
            DatasetExamplesCallback(splits=["train", "val"], n=20, random_idxs=True),
            ModelSummary(depth=5),
            SaveModelCheckpoint(name="best", metric="loss", last=True, mode="min", stage="val"),
            ArtifactsLoggerCallback(),
        ]
        return callbacks

    def create_logger(self, file_log: logging.Logger) -> Loggers:
        log.info("..Creating Logger..")
        loggers = [
            MLFlowLogger(
                self.log_path,
                config=self.to_dict(),
                experiment_name=self.setup.experiment_name,
                run_name=self.setup.run_name,
                resume=True,
            )
        ]
        logger = Loggers(loggers, self.device_id, file_log)
        return logger

    @property
    def TrainerClass(self) -> Type[Trainer]:
        return Trainer

    def create_trainer(self) -> Trainer:
        log.info("..Creating Trainer..")
        try:
            callbacks = self.create_callbacks()
            trainer = self.TrainerClass(
                logger=self.logger,
                callbacks=callbacks,
                **self.trainer.to_dict(),
            )
            return trainer
        except Exception as e:
            log.error(str(e))
            self.logger.finalize(Status.FAILED)
            raise e

    @abstractmethod
    def create_inference_model(self, device: str = "cuda:0") -> BaseInferenceModel:
        raise NotImplementedError()


if __name__ == "__main__":
    from src.utils.config import YAML_EXP_PATH

    cfg = BaseConfig.from_yaml(str(YAML_EXP_PATH / "classification/hrnet_32.yaml"))
    print(cfg)
