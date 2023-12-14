from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer

import math
import time
import torch
from abc import abstractmethod

from src.logging import get_pylogger
from src.utils.files import save_txt_to_file, save_yaml


from .results import BaseResult
from .visualization import plot_metrics


log = get_pylogger(__name__)


class BaseCallback:
    @abstractmethod
    def log(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_fit_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_train_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_train_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer):
        pass

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass


class Callbacks:
    def __init__(self, callbacks: list[BaseCallback]):
        self.callbacks = callbacks

    def log(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.log(trainer)

    def on_fit_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_train_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_train_epoch_start(trainer)

    def on_train_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer)

    def on_validation_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_epoch_start(trainer)

    def on_validation_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_epoch_end(trainer)

    def on_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    @abstractmethod
    def state_dict(self):
        state_dict = {}
        for callback in self.callbacks:
            state_dict.update(callback.state_dict())
        return state_dict

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        for callback in self.callbacks:
            callback.load_state_dict(state_dict)


class BaseExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, name: str | None, stages: list[str] | str):
        if name is None:
            name = ""
        else:
            name = "_" + name
        self.name = name
        if isinstance(stages, str):
            stages = [stages]
        self.stages = stages

    @abstractmethod
    def plot_example_results(
        self, trainer: Trainer, results: BaseResult, filepath: str
    ):
        raise NotImplementedError()

    def plot(self, trainer: Trainer, prefix: str, on_step: bool) -> None:
        if on_step:
            dirpath = trainer.logger.steps_examples_dir
        else:
            dirpath = trainer.logger.epochs_examples_dir
        for stage in self.stages:
            if stage not in trainer.module.results:
                log.warn(
                    f"{__file__}: {stage} results not yet logged (epoch={trainer.current_epoch}, step={trainer.current_step})"
                )
                continue
            results = trainer.module.results[stage]
            filepath = f"{dirpath}/{stage}/{prefix}{self.name}.jpg"
            self.plot_example_results(trainer, results, filepath)

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"epoch_{trainer.current_epoch}", on_step=False)

    def log(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"step_{trainer.current_step}", on_step=True)


class SaveModelCheckpoint(BaseCallback):
    def __init__(
        self,
        name: str | None = None,
        stage: str | None = None,
        metric: str | None = None,
        mode: str | None = "min",
        last: bool = False,
        top_k: int = 1,
        verbose: bool = False,
    ):
        self.name = name if name is not None else "best"
        self.stage = stage
        self.metric = metric
        self.save_last = last
        self.top_k = top_k
        self.verbose = verbose

        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def on_validation_epoch_end(self, trainer: Trainer):
        ckpt_dir = trainer.logger.ckpt_dir
        if self.metric is not None and self.stage is not None:
            metrics_storage = trainer.module.epochs_metrics_storage
            stage_metric_values = metrics_storage.get(self.metric, self.stage)
            if len(stage_metric_values) == 0:
                raise ValueError(
                    f"{self.metric} not yet logged to metrics storage. Current logged metrics: {metrics_storage.logged_metrics}"
                )
            last = stage_metric_values[-1]
            if self.compare(last, self.best) and self.top_k == 1:
                self.best = last
                log.info(f"Found new best value for {self.metric} ({self.stage})")
                trainer.save_checkpoint(str(ckpt_dir / f"{self.name}.pt"))
        if self.save_last:
            name = self.name if self.name is not None else "last"
            trainer.save_checkpoint(str(ckpt_dir / f"{name}.pt"))

    def state_dict(self) -> dict:
        return {f"best_{self.stage}_{self.metric}": self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict[f"best_{self.stage}_{self.metric}"]


class LoadModelCheckpoint(BaseCallback):
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    def on_fit_start(self, trainer: Trainer):
        trainer.load_checkpoint(self.ckpt_path)


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def plot(self, trainer: Trainer, on_step: bool) -> None:
        module = trainer.module
        prefix = "step" if on_step else "epoch"
        filepath = f"{trainer.logger.log_path}/{prefix}_metrics.jpg"
        storage = (
            module.steps_metrics_storage if on_step else module.epochs_metrics_storage
        )
        if len(storage.metrics) > 0:
            plot_metrics(storage, filepath=filepath)
        else:
            log.warn("No metrics to plot logged yet")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, on_step=False)

    def log(self, trainer: Trainer) -> None:
        self.plot(trainer, on_step=True)


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def save(self, trainer: Trainer, on_step: bool) -> None:
        module = trainer.module
        prefix = "step" if on_step else "epoch"
        filepath = filepath = f"{trainer.logger.log_path}/{prefix}_metrics.yaml"

        storage = (
            module.steps_metrics_storage if on_step else module.epochs_metrics_storage
        )
        if len(storage.metrics) > 0:
            metrics = storage.to_dict()
            save_yaml(metrics, filepath)
        else:
            log.warn("No metrics to save logged yet")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.save(trainer, on_step=False)

    def log(self, trainer: Trainer) -> None:
        self.save(trainer, on_step=True)


class ModelSummary(BaseCallback):
    def __init__(self, depth: int = 6):
        self.depth = depth

    def on_fit_start(self, trainer: Trainer):
        model = trainer.module.model
        model_summary = model.summary(self.depth)
        filepath = f"{trainer.logger.model_dir}/model_summary.txt"
        save_txt_to_file(model_summary, filepath)


# TODO: replace with epochs
class SaveLastAsOnnx(BaseCallback):
    def __init__(self, every_n_minutes: int = 30):
        self.every_n_minutes = every_n_minutes
        self.start_time = time.time()
        self.num_saved = 0

    def on_fit_start(self, trainer: Trainer):
        model = trainer.module.model
        dirpath = str(trainer.logger.model_onnx_dir)
        log.info("Saving model to onnx")
        filepath = f"{dirpath}/model.onnx"
        model.export_to_onnx(filepath)

    def on_validation_epoch_end(self, trainer: Trainer):
        model = trainer.module.model
        dirpath = str(trainer.logger.model_onnx_dir)
        filepath = f"{dirpath}/model.onnx"
        curr_time = time.time()
        diff_s = curr_time - self.start_time
        diff_min = math.ceil(diff_s / 60)
        if diff_min / self.every_n_minutes > 1 or self.num_saved == 0:
            self.start_time = curr_time
            log.info(
                f"{diff_min} minutes have passed. Saving model components to ONNX."
            )
            model.export_to_onnx(filepath)
            self.num_saved += 1
