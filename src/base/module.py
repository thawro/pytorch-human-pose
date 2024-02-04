from abc import abstractmethod

import torch
from torch import Tensor, optim
from torch.nn.modules.loss import _Loss

from src.logging import get_pylogger
from src.logging.loggers import BaseLogger

from .datamodule import DataModule

from .model import BaseModel
from .callbacks import Callbacks
from .lr_scheduler import LRScheduler

log = get_pylogger(__name__)

SPLITS = ["train", "val", "test"]


class BaseModule:
    model: BaseModel
    logger: BaseLogger
    device: torch.device
    datamodule: DataModule
    callbacks: "Callbacks"
    current_epoch: int
    current_step: int
    log_every_n_steps: int
    limit_batches: int
    optimizers: dict[str, optim.Optimizer]
    schedulers: dict[str, LRScheduler]

    def __init__(self, model: BaseModel, loss_fn: _Loss, use_fp16: bool = True):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.current_epoch = 0
        self.current_step = 0
        self.use_fp16 = use_fp16
        self.optimizers, self.schedulers = self.create_optimizers()

    def pass_attributes(
        self,
        device_id: int,
        device: torch.device,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
        limit_batches: int,
        log_every_n_steps: int,
    ):
        self.device_id = device_id
        self.logger = logger
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.limit_batches = limit_batches
        self.total_batches = datamodule.total_batches
        if limit_batches > 0:
            self.total_batches = {k: limit_batches for k in self.total_batches}
        self.device = device
        self.log_every_n_steps = log_every_n_steps

    def set_attributes(self, **attributes):
        for name, attr in attributes.items():
            setattr(self, name, attr)

    def load_state_dict(self, state_dict: dict, lr: float | None = None):
        self.model.load_state_dict(state_dict["model"])
        self.optimizers, self.schedulers = self.create_optimizers()
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
            if lr is not None:
                optimizer.param_groups[0]["lr"] = lr
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state_dict["schedulers"][name])

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }

        model_state = {"model": self.model.state_dict()}
        model_state.update(
            {
                "optimizers": optimizers_state,
                "schedulers": schedulers_state,
            }
        )
        return model_state

    @abstractmethod
    def _common_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, float]:
        raise NotImplementedError()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, float]:
        self.stage = "train"
        self.model.train()
        metrics = self._common_step(batch, batch_idx)
        for name, scheduler in self.schedulers.items():
            if scheduler.is_step_interval:
                scheduler.step()
        return metrics

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> dict[str, float]:
        self.stage = stage
        with torch.no_grad():
            self.model.eval()
            metrics = self._common_step(batch, batch_idx)
        return metrics

    def on_epoch_end(self) -> None:
        for name, scheduler in self.schedulers.items():
            if scheduler.is_epoch_interval:
                scheduler.step()

    @property
    def is_log_step(self):
        return self.current_step % self.log_every_n_steps == 0

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        raise NotImplementedError()
