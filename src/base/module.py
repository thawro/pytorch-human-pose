from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from src.logging import get_pylogger
from src.logging.loggers import BaseLogger

from .datamodule import DataModule
from .storage import MetricsStorage
from .results import BaseResult
from .model import BaseModel
from .callbacks import Callbacks

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

    def __init__(
        self,
        model: BaseModel,
        loss_fn: _Loss,
        optimizers: dict[str, torch.optim.Optimizer],
        schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {},
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.steps_metrics = MetricsStorage(name="Steps")  # every step metrics
        self.validation_metrics = MetricsStorage(name="LogStep")  # validation metrics
        self.current_epoch = 0
        self.current_step = 0

        self.results: dict[str, BaseResult] = {}

    def pass_attributes(
        self,
        device: torch.device,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
        limit_batches: int,
        log_every_n_steps: int,
    ):
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
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
            if lr is not None:
                optimizer.param_groups[0]["lr"] = lr
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state_dict["schedulers"][name])
        self.steps_metrics.load_state_dict(state_dict["metrics"]["steps"])
        self.validation_metrics.load_state_dict(state_dict["metrics"]["validation"])

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        metrics_state = {
            "steps": self.steps_metrics.state_dict(),
            "validation": self.validation_metrics.state_dict(),
        }
        model_state = {"model": self.model.state_dict()}
        model_state.update(
            {
                "optimizers": optimizers_state,
                "schedulers": schedulers_state,
                "metrics": metrics_state,
            }
        )
        return model_state

    @abstractmethod
    def _common_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        raise NotImplementedError()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        self.stage = "train"
        self.model.train()
        self._common_step(batch, batch_idx)
        for name, scheduler in self.schedulers.items():
            scheduler.step()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str):
        self.stage = stage
        self.model.eval()
        with torch.no_grad():
            self._common_step(batch, batch_idx)

    def on_epoch_end(self) -> None:
        epochs_metrics = self.steps_metrics.aggregate_over_key(
            key="epoch"
        ).inverse_nest()
        for stage, metrics in epochs_metrics.items():
            last_epoch_metrics = {
                name: values[-1]["value"] for name, values in metrics.items()
            }
            msg = [f"Epoch: {self.current_epoch}"]
            for name, value in last_epoch_metrics.items():
                msg.append(f"{stage}/{name}: {round(value, 3)}")
            log.info("  ".join(msg))

    def log_optimizer_params(self):
        optizers_lr = {
            f"{name}_LR": optimizer.param_groups[0]["lr"]
            for name, optimizer in self.optimizers.items()
        }
        self.steps_metrics.append(
            optizers_lr, self.current_step, self.current_epoch, split="train"
        )

    @property
    def is_log_step(self):
        return self.current_step % self.log_every_n_steps == 0
