from abc import abstractmethod

import torch
from torch import Tensor, optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP

from src.logger.loggers import BaseLogger
from src.logger.pylogger import log

from .callbacks import Callbacks
from .datamodule import DataModule
from .lr_scheduler import LRScheduler
from .model import BaseModel

SPLITS = ["train", "val", "test"]


class BaseModule:
    device: str
    device_id: int
    use_distributed: bool
    use_fp16: bool
    model: BaseModel
    logger: BaseLogger
    datamodule: DataModule
    callbacks: "Callbacks"
    current_epoch: int
    current_step: int
    log_every_n_steps: int
    limit_batches: int
    optimizers: dict[str, optim.Optimizer]
    schedulers: dict[str, LRScheduler]
    scalers: dict[str, GradScaler]

    def __init__(self, model: BaseModel, loss_fn: _Loss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.current_epoch = 0
        self.current_step = 0

    def set_optimizers(self):
        self.optimizers, self.schedulers = self.create_optimizers()
        self.scalers = {name: GradScaler() for name in self.optimizers}

    def to_DDP(self, device_id: int):
        self.model.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model.net)
        self.model.net = DDP(
            self.model.net.cuda(device_id),
            device_ids=[device_id],  # , find_unused_parameters=True
        )

    def compile(self):
        log.info("Compiling Module (`torch.compile(net)`)")
        self.model.net = torch.compile(self.model.net)

    # def to_fp16(self):
    #     self.model.net = network_to_half(self.model.net)

    def pass_attributes(
        self,
        device_id: int,
        device: str,
        use_distributed: bool,
        use_fp16: bool,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
        limit_batches: int,
        log_every_n_steps: int,
    ):
        self.device_id = device_id
        self.device = device
        self.use_distributed = use_distributed
        self.use_fp16 = use_fp16
        self.logger = logger
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.limit_batches = limit_batches
        self.total_batches = datamodule.total_batches
        if limit_batches > 0:
            self.total_batches = {k: limit_batches for k in self.total_batches}
        self.log_every_n_steps = log_every_n_steps

    def set_attributes(self, **attributes):
        for name, attr in attributes.items():
            setattr(self, name, attr)

    def load_state_dict(self, state_dict: dict, lr: float | None = None):
        model_state_dict = state_dict["model"]
        model_state_dict = self.model.parse_checkpoint(model_state_dict)
        self.model.load_state_dict(model_state_dict)
        self.set_optimizers()
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
            if lr is not None:
                optimizer.param_groups[0]["lr"] = lr
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state_dict["schedulers"][name])
        for name, scaler in self.scalers.items():
            scaler.load_state_dict(state_dict["scalers"][name])

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        scalers_state = {name: scaler.state_dict() for name, scaler in self.scalers.items()}

        model_state = {"model": self.model.state_dict()}
        model_state.update(
            {
                "optimizers": optimizers_state,
                "schedulers": schedulers_state,
                "scalers": scalers_state,
            }
        )
        return model_state

    @abstractmethod
    def _common_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        raise NotImplementedError()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        self.stage = "train"
        metrics = self._common_step(batch, batch_idx)
        for name, scheduler in self.schedulers.items():
            if scheduler.is_step_interval:
                scheduler.step()
        return metrics

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> dict[str, float]:
        self.stage = stage
        metrics = self._common_step(batch, batch_idx)
        return metrics

    def on_epoch_start(self) -> None:
        pass

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
