from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

from PIL import Image

if TYPE_CHECKING:
    from src.base.storage import MetricsStorage

    from .trainer import Trainer

import glob
from abc import abstractmethod
from pathlib import Path

import torch

from src.logger.loggers import Status
from src.logger.monitoring.system import SystemMetricsMonitor
from src.logger.pylogger import log
from src.utils.config import LOG_DEVICE_ID
from src.utils.files import save_txt_to_file, save_yaml

from .results import BaseResult, plot_results
from .visualization import plot_metrics_matplotlib, plot_metrics_plotly, plot_system_monitoring

_log_mode = Literal["step", "epoch", "validation"]


def get_metrics_storage(trainer: Trainer, mode: _log_mode) -> MetricsStorage:
    if mode == "validation":
        return trainer.validation_metrics.aggregate_over_key(key="step")
    elif mode == "epoch":
        return trainer.epochs_metrics.aggregate_over_key(key="epoch")
    else:
        raise ValueError("Wrong logging mode")


class BaseCallback:
    @abstractmethod
    def on_fit_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_failure(self, trainer: Trainer, status: Status):
        pass

    @abstractmethod
    def on_step_end(self, trainer: Trainer):
        pass

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass


class Callbacks:
    def __init__(self, callbacks: list[BaseCallback], device_id: int):
        # make sure that only device at LOG_DEVICE_ID is callbacking
        if device_id != LOG_DEVICE_ID:
            callbacks = []
        self.callbacks = callbacks
        self.device_id = device_id

    def on_step_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_step_end(trainer)

    def on_fit_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_validation_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_start(trainer)

    def on_validation_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_end(trainer)

    def on_failure(self, trainer: Trainer, status: Status):
        log.warn("Failure mode detected. Running callbacks `on_failure` methods")
        for callback in self.callbacks:
            callback.on_failure(trainer, status)

    def state_dict(self) -> dict[str, dict]:
        state_dict = {}
        for callback in self.callbacks:
            name = callback.__class__.__name__
            state_dict[name] = callback.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, dict]):
        for callback in self.callbacks:
            name = callback.__class__.__name__
            callback.load_state_dict(state_dict.get(name, {}))


class ArtifactsLoggerCallback(BaseCallback):
    """Call logger artifacts logging methods"""

    def on_fit_start(self, trainer: Trainer) -> None:
        data_examples_dir = str(trainer.logger.loggers[0].data_examples_dir)
        trainer.logger.log_artifacts(data_examples_dir, "data_examples")

    def on_step_end(self, trainer: Trainer) -> None:
        """Log logs files"""
        trainer.logger.log_logs()

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Log artifacts directories and/or files"""
        eval_examples_dir = str(trainer.logger.loggers[0].eval_examples_dir)
        trainer.logger.log_artifacts(eval_examples_dir, "eval_examples")

        log_dir = str(trainer.logger.loggers[0].log_path)
        metrics_filepaths = glob.glob(f"{log_dir}/epoch_metrics.*")
        for filepath in metrics_filepaths:
            trainer.logger.log_artifact(filepath, "epoch_metrics")
        log.info("Artifacts logged to remote ('eval_examples' and 'epoch_metrics').")

    def on_failure(self, trainer: Trainer, status: Status):
        log.warn("Finalizing loggers.")
        trainer.logger.log_logs()
        trainer.logger.finalize(status=status)


class SaveModelCheckpoint(BaseCallback):
    def __init__(
        self,
        name: str | None = None,
        stage: str | None = None,
        metric: str | None = None,
        mode: str | None = "min",
        last: bool = False,
        verbose: bool = False,
    ):
        self.name = name if name is not None else "best"

        self.stage = stage
        self.metric = metric
        self.save_last = last
        self.verbose = verbose
        self.mode = mode

        self.callback_name = f"{metric if metric else ''} {self.name} ({mode})"

        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def save_model(self, trainer: Trainer):
        ckpt_dir = trainer.logger.loggers[0].ckpt_dir

        if self.metric is not None and self.stage is not None:
            meters = trainer.meters[self.stage]
            if len(meters) == 0:
                e = ValueError(f"{self.metric} not yet logged to meters")
                log.exception(e)
                raise e
            last = meters[self.metric].avg
            log.info(
                f"Current {self.stage}/{self.metric}={last:.3e},   Best {self.stage}/{self.metric}={self.best:.3e}"
            )
            if self.compare(last, self.best):
                self.best = last
                log.info(f"Found new best value for {self.stage}/{self.metric} ({self.best:.3e})")
                trainer.save_checkpoint(str(ckpt_dir / f"{self.name}.pt"))
        if self.save_last:
            filename = "last"
            # f"epoch_{trainer.current_epoch}_step_{trainer.current_step}"
            trainer.save_checkpoint(str(ckpt_dir / f"{filename}.pt"))

    def on_epoch_end(self, trainer: Trainer):
        self.save_model(trainer)

    @property
    def state_metric_name(self) -> str:
        return f"best_{self.stage}_{self.metric}"

    def state_dict(self) -> dict:
        return {self.state_metric_name: self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict.get(self.state_metric_name, self.best)
        log.info(
            f'     Loaded "{self.__class__.__name__}" state ({self.state_metric_name} = {self.best})'
        )


class ResultsPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, name: str):
        self.name = name

    def plot_example_results(self, trainer: Trainer, results: list[BaseResult], filepath: str):
        plot_results(results, plot_name=self.name, filepath=filepath)

    def plot(self, trainer: Trainer, prefix: str) -> None:
        dirpath = trainer.logger.loggers[0].eval_examples_dir
        dirpath.mkdir(exist_ok=True, parents=True)
        if self.name != "":
            prefix = f"{prefix}_"
        filepath = str(dirpath / f"{prefix}{self.name}.jpg")
        if len(trainer.results) > 0:
            self.plot_example_results(trainer, trainer.results, filepath)
            log.info(f"{self.name.capitalize()} results visualization saved at '{filepath}'")
        else:
            log.warn("No results to visualize")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"validation_{trainer.current_step}")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"epoch_{trainer.current_epoch}")


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def plot(self, trainer: Trainer, mode: _log_mode) -> None:
        log_path = trainer.logger.loggers[0].log_path

        storage = get_metrics_storage(trainer, mode)
        if len(storage.metrics) > 0:
            step_name = "epoch" if mode == "epoch" else "step"
            mpl_filepath = f"{log_path}/{mode}_metrics.jpg"
            plot_metrics_matplotlib(storage, step_name, filepath=mpl_filepath)

            plotly_filepath = f"{log_path}/{mode}_metrics.html"
            plot_metrics_plotly(storage, step_name, filepath=plotly_filepath)
            log.info(f"Metrics plots saved at '{plotly_filepath}'")
        else:
            log.warn(f"No metrics to plot logged yet (mode={mode})")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="validation")


class SystemMetricsMonitoringCallback(BaseCallback):
    """Plot per epoch metrics"""

    def __init__(self, sampling_interval: int = 10, samples_before_logging: int = 1) -> None:
        self.sampling_interval = sampling_interval
        self.samples_before_logging = samples_before_logging

    def plot(self, trainer: Trainer) -> None:
        log_path = trainer.logger.loggers[0].log_path
        filepath = f"{log_path}/logs/system_metrics.html"

        storage = trainer.system_monitoring
        if len(storage.metrics) > 0:
            plot_system_monitoring(storage, filepath)

    def on_fit_start(self, trainer: Trainer):
        system_monitor = SystemMetricsMonitor(
            sampling_interval=self.sampling_interval,
            samples_before_logging=self.samples_before_logging,
            metrics_callback=trainer.system_monitoring.update,
        )
        system_monitor.start()

    def on_step_end(self, trainer: Trainer) -> None:
        self.plot(trainer)


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def save(self, trainer: Trainer, mode: _log_mode) -> None:
        storage = get_metrics_storage(trainer, mode)
        log_path = trainer.logger.loggers[0].log_path
        filepath = filepath = f"{log_path}/{mode}_metrics.yaml"

        if len(storage.metrics) > 0:
            metrics = storage.to_dict()
            save_yaml(metrics, filepath)
        else:
            log.warn(f"No metrics to save logged yet (mode={mode})")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="validation")


class MetricsLogger(BaseCallback):
    """Log per epoch metrics to terminal"""

    def on_epoch_end(self, trainer: Trainer) -> None:
        msgs = ""
        for stage, metrics in trainer.epochs_metrics.inverse_nest().items():
            last_epoch_metrics = {name: values[-1]["value"] for name, values in metrics.items()}
            msg = []
            for name, value in last_epoch_metrics.items():
                msg.append(f"{stage}/{name}: {value:.3e}")
            msg = "  ".join(msg)
            msgs += f"     {msg}\n"
        msgs = msgs[:-2]  # remove last line break
        log.info(f"Epoch {trainer.current_epoch} metrics:\n{msgs}")


class ModelSummary(BaseCallback):
    def __init__(self, depth: int = 6):
        self.depth = depth

    def on_fit_start(self, trainer: Trainer):
        log.info(f"Optimizers\n{trainer.module.optimizers}")
        log.info(f"LR Schedulers\n{trainer.module.lr_schedulers}")

        model = trainer.module.model
        model_summary = model.summary(self.depth)
        log.info(f"Model layers summary\n{model_summary}")
        model_dir = trainer.logger.loggers[0].model_dir
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        filepath = f"{model_dir}/model_summary.txt"
        save_txt_to_file(model_summary, filepath)


class DatasetExamplesCallback(BaseCallback):
    def __init__(
        self,
        splits: list[Literal["train", "val", "test"]] = ["train"],
        n: int = 10,
        random_idxs: bool = False,
    ) -> None:
        self.splits = splits
        self.n = n
        self.random_idxs = random_idxs

    def on_fit_start(self, trainer: Trainer):
        log.info("..Saving datasets samples examples..")
        dirpath = trainer.logger.loggers[0].data_examples_dir
        for split in self.splits:
            dataset = trainer.datamodule.datasets[split]
            dirpath.mkdir(exist_ok=True, parents=True)
            filepath = str(dirpath / f"{split}.jpg")

            if self.random_idxs:
                idxs = [random.randint(0, len(dataset)) for _ in range(self.n)]
            else:
                idxs = list(range(self.n))
            grid = dataset.plot_examples(idxs, nrows=1)
            Image.fromarray(grid).save(filepath)
            log.info(f"      ..{split} dataset examples saved at '{filepath}'..")
