from __future__ import annotations
from typing import TYPE_CHECKING
from src.metrics.results import Result

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from abc import abstractmethod
from src.logging import get_pylogger

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
    def plot_example_results(self, trainer: Trainer, results: Result, filepath: str):
        raise NotImplementedError()

    def plot(self, trainer: Trainer, prefix: str) -> None:
        for stage in self.stages:
            if stage not in trainer.module.results:
                log.warn(
                    f"{__file__}: {stage} results not yet logged (epoch={trainer.current_epoch}, step={trainer.current_step})"
                )
                continue
            results = trainer.module.results[stage]
            filepath = f"{trainer.logger.examples_dir}/{stage}/{prefix}{self.name}.jpg"
            self.plot_example_results(trainer, results, filepath)

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"epoch_{trainer.current_epoch}")

    def log(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"step_{trainer.current_step}")
