import torch
import math
from typing import Literal
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
import torch.distributed as dist

from .datamodule import DataModule
from .module import BaseModule
from .callbacks import BaseCallback, Callbacks
from .meters import Meters
from .storage import MetricsStorage
import random

log = get_pylogger(__name__)


_stage = Literal["train", "val", "eval_val"]


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: BaseLogger,
        device_id: int,
        callbacks: list[BaseCallback],
        max_epochs: int = 100,
        limit_batches: int = -1,
        log_every_n_steps: int = -1,
        use_distributed: bool = True,
        use_fp16: bool = True,
    ):
        stages = ["train", "val", "eval_val"]
        self.use_distributed = use_distributed
        self.use_fp16 = use_fp16
        self.meters = {
            stage: Meters(use_distributed=use_distributed) for stage in stages
        }
        self.logger = logger
        self.device = f"cuda:{device_id}"
        self.device_id = device_id
        self.callbacks = Callbacks(callbacks, device_id=device_id)
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.log_every_n_steps = log_every_n_steps
        self.current_step = 0
        self.current_epoch = 0
        self.epochs_metrics = MetricsStorage(name="Epochs")  # every step metrics
        self.validation_metrics = MetricsStorage(name="LogStep")  # validation metrics
        self.results = []

    def batch_to_device(self, batch) -> None:
        for j in range(len(batch)):
            if isinstance(batch[j], Tensor):
                batch[j] = batch[j].to(self.device)
            elif isinstance(batch[j], list):  # list of tensors
                batch[j] = [batch[j][i].to(self.device) for i in range(len(batch[j]))]
            elif isinstance(batch[j], dict):  # dict of tensors
                batch[j] = {
                    key: value.to(self.device) if isinstance(value, Tensor) else value
                    for key, value in batch[j].items()
                }

    def get_limit_batches(self) -> int:
        if self.current_step == 0:
            return 1
        else:
            return int(self._limit_batches)

    def _update_metrics(self, stages: list[_stage]):
        if "eval_val" in stages:
            storage = self.validation_metrics
        else:
            storage = self.epochs_metrics
        if "train" in stages:
            optizers_lr = {
                f"{name}_LR": optimizer.param_groups[0]["lr"]
                for name, optimizer in self.module.optimizers.items()
            }
            storage.append(
                optizers_lr, self.current_step, self.current_epoch, split="train"
            )
        for stage in stages:
            metrics = {
                name: meter.avg for name, meter in self.meters[stage].meters.items()
            }
            storage.append(metrics, self.current_step, self.current_epoch, stage)

    def evaluate(self, dataloader: DataLoader, stage: _stage):
        """Evaluate on validation set"""
        meters = self.meters[stage]
        meters.reset()
        n_batches = len(dataloader)
        random_idx = random.randint(0, n_batches - 1)

        with torch.no_grad():
            loop = tqdm(dataloader, leave=True, desc=stage)
            limit_batches = self.get_limit_batches()
            for i, batch in enumerate(loop):
                self.batch_to_device(batch)
                val_metrics, val_results = self.module.validation_step(
                    batch, i, stage=stage
                )
                meters.update(val_metrics, batch[0].shape[0])
                if stage == "eval_val" and i == random_idx:
                    self.results.extend(val_results)
                limit_batches -= 1
                if limit_batches == 0:
                    break
        meters.all_reduce()

    def sanity_check(self, dataloader: DataLoader):
        """Run sanity check"""
        loop = tqdm(dataloader, leave=True, desc="Sanity check")
        limit_batches = 1
        for i, batch in enumerate(loop):
            self.batch_to_device(batch)
            self.module.validation_step(batch, i, stage="sanity_check")
            limit_batches -= 1
            if limit_batches == 0:
                break

    def single_epoch(self):
        meters = self.meters["train"]
        meters.reset()

        loop = tqdm(self.datamodule.train_dataloader, leave=True, desc="Train")
        limit_batches = int(self._limit_batches)
        for i, batch in enumerate(loop):
            self.batch_to_device(batch)
            train_metrics = self.module.training_step(batch, i)
            meters.update(train_metrics, batch[0].shape[0])
            self.current_step += 1
            self.module.set_attributes(current_step=self.current_step)
            if self.current_step % self.log_every_n_steps == 0:
                self.callbacks.on_validation_start(self)
                self.evaluate(self.datamodule.val_dataloader, stage="eval_val")
                self._update_metrics(stages=["eval_val"])
                self.callbacks.on_validation_end(self)
                self.results = []

            limit_batches -= 1
            if limit_batches == 0:
                break
        meters.all_reduce()

    def fit(
        self, module: BaseModule, datamodule: DataModule, ckpt_path: str | None = None
    ):
        n_train_batches = datamodule.total_batches["train"] - 1
        limit_batches = (
            self._limit_batches if self._limit_batches > 0 else n_train_batches
        )
        if self.log_every_n_steps < 0:
            self.log_every_n_steps = abs(self.log_every_n_steps) * min(
                n_train_batches, limit_batches
            )
        self.datamodule = datamodule

        self.module = module
        self.module.pass_attributes(
            device_id=self.device_id,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            datamodule=datamodule,
            limit_batches=int(self._limit_batches),
            log_every_n_steps=self.log_every_n_steps,
        )

        self.callbacks.on_fit_start(self)
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        self.module.set_attributes(current_step=self.current_step)
        self.sanity_check(self.datamodule.val_dataloader)
        try:
            for epoch in range(
                self.current_epoch, self.current_epoch + self.max_epochs
            ):
                if self.use_distributed:
                    self.datamodule.train_dataloader.sampler.set_epoch(epoch)  # DDP
                self.current_epoch = epoch
                self.callbacks.on_epoch_start(self)
                module.set_attributes(current_epoch=epoch)
                self.single_epoch()
                self.evaluate(self.datamodule.val_dataloader, stage="val")
                self._update_metrics(stages=["train", "val"])
                self.module.on_epoch_end()
                self.callbacks.on_epoch_end(self)
                log.info(f" <<<  {self.device_info} epoch finished  >>> ")
                print(self.use_distributed)
                if self.use_distributed:
                    dist.barrier()
        except KeyboardInterrupt as e:
            self.callbacks.on_failure(self)
            raise e

    def load_checkpoint(self, ckpt_path: str, lr: float | None = None):
        log.info(f"{self.device_info}Loading checkpoint from {ckpt_path}")
        map_location = {"cuda:0": self.device}

        ckpt_state = torch.load(ckpt_path, map_location=map_location)
        self.epochs_metrics.load_state_dict(ckpt_state["metrics"]["steps"])
        self.validation_metrics.load_state_dict(ckpt_state["metrics"]["validation"])
        self.module.load_state_dict(ckpt_state["module"], lr=lr)
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.current_step = ckpt_state["step"]

        n_train_batches = self.datamodule.total_batches["train"] - 1
        limit_batches = (
            self._limit_batches if self._limit_batches > 0 else n_train_batches
        )
        self.current_epoch = math.ceil(self.current_step / limit_batches)
        log.info(
            f"{self.device_info}The training is resumed at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )

    @property
    def device_info(self) -> str:
        return f"[{self.device}] "

    def save_checkpoint(self, ckpt_path: str):
        log.info(f"{self.device_info}Saving checkpoint to {ckpt_path}")
        if self.device_id != 0:  # save only for cuda:0 (DDP)
            return
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        callbacks_state = self.callbacks.state_dict()
        metrics_state = {
            "steps": self.epochs_metrics.state_dict(),
            "validation": self.validation_metrics.state_dict(),
        }

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "metrics": metrics_state,
            "callbacks": callbacks_state,
            "epoch": self.current_epoch,
            "step": self.current_step,
        }
        torch.save(ckpt_state, ckpt_path)
        log.info(
            f"{self.device_info}"
            f"The training is saved at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}",
        )
