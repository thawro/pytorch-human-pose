from __future__ import annotations
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from .trainer import Trainer
    from src.base.module import BaseModule
    from src.base.storage import MetricsStorage

import math
import time
import torch
from abc import abstractmethod

from src.logging import get_pylogger
from src.utils.files import save_txt_to_file, save_yaml


from .results import BaseResult
from .visualization import plot_metrics


_log_mode = Literal["step", "epoch", "validation"]


def get_metrics_storage(module: BaseModule, mode: _log_mode) -> MetricsStorage:
    if mode == "step":
        return module.steps_metrics.aggregate_over_key(key="step")
    elif mode == "validation":
        return module.validation_metrics.aggregate_over_key(key="step")
    elif mode == "epoch":
        return module.steps_metrics.aggregate_over_key(key="epoch")
    else:
        raise ValueError("Wrong logging mode")


log = get_pylogger(__name__)


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
    def on_failure(self, trainer: Trainer):
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
    def __init__(self, callbacks: list[BaseCallback]):
        self.callbacks = callbacks

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

    def on_failure(self, trainer: Trainer):
        log.warn("Failure mode detected. Running callbacks `on_failure` methods")
        for callback in self.callbacks:
            callback.on_failure(trainer)

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
        self.mode = mode

        self.callback_name = f"{metric if metric else ''} {self.name} ({mode})"

        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def save_model(self, trainer: Trainer):
        ckpt_dir = trainer.logger.ckpt_dir
        if self.save_last:
            trainer.save_checkpoint(str(ckpt_dir / f"last.pt"))
        if self.metric is not None and self.stage is not None:
            metrics = trainer.module.validation_metrics
            stage_metric_values = metrics.get(self.metric, self.stage)
            if len(stage_metric_values) == 0:
                raise ValueError(
                    f"{self.metric} not yet logged to metrics storage. Current logged metrics: {metrics.logged_metrics}"
                )
            last = stage_metric_values[-1]["value"]
            if self.compare(last, self.best) and self.top_k == 1:
                self.best = last
                log.info(f"Found new best value for {self.metric} ({self.stage})")
                trainer.save_checkpoint(str(ckpt_dir / f"{self.name}.pt"))

    def on_validation_end(self, trainer: Trainer):
        self.save_model(trainer)

    def on_failure(self, trainer: Trainer):
        log.warn(f"`on_failure`: Saving {self.callback_name}")
        self.save_model(trainer)

    def state_dict(self) -> dict:
        return {f"best_{self.stage}_{self.metric}": self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict.get(f"best_{self.stage}_{self.metric}", self.best)


class LoadModelCheckpoint(BaseCallback):
    def __init__(self, ckpt_path: str, lr: float | None = None):
        self.ckpt_path = ckpt_path
        self.lr = lr

    def on_fit_start(self, trainer: Trainer):
        trainer.load_checkpoint(self.ckpt_path, lr=self.lr)


class BaseExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, name: str | None):
        if name is None:
            name = ""
        else:
            name = "_" + name
        self.name = name

    @abstractmethod
    def plot_example_results(
        self, trainer: Trainer, results: BaseResult, filepath: str
    ):
        raise NotImplementedError()

    def plot(self, trainer: Trainer, prefix: str, mode: _log_mode) -> None:
        dirpath = trainer.logger.eval_examples_dir
        for stage, results in trainer.module.results.items():
            stage_dirpath = dirpath / stage
            stage_dirpath.mkdir(exist_ok=True, parents=True)
            filepath = stage_dirpath / f"{prefix}{self.name}.jpg"
            self.plot_example_results(trainer, results, str(filepath))

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(
            trainer, prefix=f"validation_{trainer.current_step}", mode="validation"
        )


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def plot(self, trainer: Trainer, mode: _log_mode) -> None:
        filepath = f"{trainer.logger.log_path}/{mode}_metrics.jpg"
        storage = get_metrics_storage(trainer.module, mode)
        if len(storage.metrics) > 0:
            step_name = "epoch" if mode == "epoch" else "step"
            plot_metrics(storage, step_name, filepath=filepath)
        else:
            log.warn(f"No metrics to plot logged yet (mode={mode})")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="validation")
        self.plot(trainer, mode="step")


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def save(self, trainer: Trainer, mode: _log_mode) -> None:
        storage = get_metrics_storage(trainer.module, mode)
        filepath = filepath = f"{trainer.logger.log_path}/{mode}_metrics.yaml"

        if len(storage.metrics) > 0:
            metrics = storage.to_dict()
            save_yaml(metrics, filepath)
        else:
            log.warn(f"No metrics to save logged yet (mode={mode})")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="validation")
        self.save(trainer, mode="step")


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

    def on_validation_end(self, trainer: Trainer):
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
