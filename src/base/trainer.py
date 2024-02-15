import torch
from typing import Literal
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.logging import get_pylogger

log = get_pylogger(__name__)

from src.logging.loggers import BaseLogger
import torch.distributed as dist

from .datamodule import DataModule
from .module import BaseModule
from .callbacks import BaseCallback, Callbacks
from .meters import Meters
from .storage import MetricsStorage
import random
import os


_stage = Literal["train", "val", "eval_val"]
_accelerator = Literal["cpu", "gpu"]


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: BaseLogger,
        accelerator: _accelerator,
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
        self.accelerator = accelerator
        self._set_device()
        self.callbacks = Callbacks(callbacks, device_id=self.device_id)
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.log_every_n_steps = log_every_n_steps
        self.current_step = 0
        self.current_epoch = 0
        self.epochs_metrics = MetricsStorage(name="Epochs")  # every step metrics
        self.validation_metrics = MetricsStorage(name="LogStep")  # validation metrics
        self.results = []

    @property
    def map_location(self) -> dict[str, str]:
        return {"cuda:0": self.device}

    def _set_device(self):
        if self.accelerator == "gpu":
            if self.use_distributed and "LOCAL_RANK" in os.environ:
                device_id = int(os.environ["LOCAL_RANK"])
            else:
                device_id = 0
            device = f"cuda:{device_id}"
        else:
            device_id = 0
            device = "cpu"
        self.device = device
        self.device_id = device_id

    def batch_to_device(self, batch) -> None:
        for j in range(len(batch)):
            sample = batch[j]
            if isinstance(sample, Tensor):
                batch[j] = sample.to(self.device)
            elif isinstance(sample, list):  # list of tensors
                batch[j] = [
                    (x.to(self.device) if isinstance(x, Tensor) else x) for x in sample
                ]
            elif isinstance(sample, dict):  # dict of tensors
                batch[j] = {
                    key: (value.to(self.device) if isinstance(value, Tensor) else value)
                    for key, value in sample.items()
                }

    def get_limit_batches(self, dataloader: DataLoader) -> int:
        if self.current_step == 0:
            return 1
        else:
            if self._limit_batches <= 0:
                return len(dataloader)
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
        self.module.model.net.eval()
        meters = self.meters[stage]
        meters.reset()
        limit_batches = self.get_limit_batches(dataloader)
        random_idx = random.randint(0, limit_batches - 1)

        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(dataloader, leave=True, desc=stage):
                self.batch_to_device(batch)
                val_metrics, val_results = self.module.validation_step(
                    batch, batch_idx, stage=stage
                )
                meters.update(val_metrics, batch[0].shape[0])

                if stage == "eval_val" and batch_idx == random_idx:
                    self.results.extend(val_results)
                batch_idx += 1
                limit_batches -= 1
                if limit_batches == 0:
                    break
            meters.all_reduce()

    def sanity_check(self, dataloader: DataLoader):
        """Run sanity check"""
        self.module.model.net.eval()
        loop = tqdm(dataloader, leave=True, desc="Sanity check")
        limit_batches = 1
        for i, batch in enumerate(loop):
            self.batch_to_device(batch)
            self.module.validation_step(batch, i, stage="sanity_check")
            limit_batches -= 1
            if limit_batches == 0:
                break

    def single_epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.module.model.net.train()
        meters = self.meters["train"]
        meters.reset()

        limit_batches = int(self._limit_batches)
        batch_idx = 0
        for batch in tqdm(train_dataloader, leave=True, desc="Train"):
            self.batch_to_device(batch)
            train_metrics = self.module.training_step(batch, batch_idx)
            meters.update(train_metrics, batch[0].shape[0])
            self.current_step += 1
            batch_idx += 1
            self.module.set_attributes(current_step=self.current_step)
            # if (
            #     self.log_every_n_steps != 0
            #     and self.current_step % self.log_every_n_steps == 0
            # ):
            #     self.callbacks.on_validation_start(self)
            #     self.evaluate(val_dataloader, stage="eval_val")
            #     self.module.model.train()
            #     self._update_metrics(stages=["eval_val"])
            #     self.callbacks.on_validation_end(self)
            #     self.results.clear()

            limit_batches -= 1
            if limit_batches == 0:
                break
        meters.all_reduce()

    def fit(
        self,
        module: BaseModule,
        datamodule: DataModule,
        pretrained_ckpt_path: str | None = None,
        ckpt_path: str | None = None,
    ):
        train_dataloader = datamodule.train_dataloader(self.use_distributed)
        val_dataloader = datamodule.val_dataloader(self.use_distributed)

        n_train_batches = datamodule.total_batches["train"] - 1
        limit_batches = (
            self._limit_batches if self._limit_batches > 0 else n_train_batches
        )
        if self.log_every_n_steps < 0:
            self.log_every_n_steps = abs(self.log_every_n_steps) * min(
                n_train_batches, limit_batches
            )

        if self.use_distributed:
            self.log_info("Moving to DDP (Data Distributed Parallel)")
            module.to_DDP(self.device_id)
        else:
            module.model.cuda(self.device_id)
        module.loss_fn.cuda(self.device_id)

        if self.use_fp16:
            self.log_info("Changing to FP16")
            module.to_fp16()

        module.pass_attributes(
            device_id=self.device_id,
            device=self.device,
            use_distributed=self.use_distributed,
            use_fp16=self.use_fp16,
            logger=self.logger,
            callbacks=self.callbacks,
            datamodule=datamodule,
            limit_batches=int(self._limit_batches),
            log_every_n_steps=self.log_every_n_steps,
        )
        self.module = module
        self.callbacks.on_fit_start(self)
        self.log_info("Compiling Module (`torch.compile(net)`)")
        self.module.compile()
        self.log_info(
            "Initializing weights and [optionally] loading pretrained weights"
        )
        self.module.model.init_weights(pretrained_ckpt_path, self.map_location)
        self.module.set_optimizers()
        self.datamodule = datamodule

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        self.module.set_attributes(current_step=self.current_step)
        self.sanity_check(val_dataloader)
        try:
            for epoch in range(
                self.current_epoch, self.current_epoch + self.max_epochs
            ):
                if self.use_distributed:
                    train_dataloader.sampler.set_epoch(epoch)  # DDP
                self.current_epoch = epoch
                self.module.on_epoch_start()
                self.callbacks.on_epoch_start(self)
                module.set_attributes(current_epoch=epoch)
                self.single_epoch(train_dataloader, val_dataloader)
                self.evaluate(val_dataloader, stage="val")
                self._update_metrics(stages=["train", "val"])
                self.module.on_epoch_end()
                self.callbacks.on_epoch_end(self)
                self.results.clear()
                self.log_info(f" <<<  epoch {epoch} finished  >>> ")
                if self.use_distributed:
                    dist.barrier()
        except KeyboardInterrupt as e:
            self.callbacks.on_failure(self)
            raise e

    def log_info(self, msg: str) -> None:
        log.info(f"{self.device_info}{msg}")

    def load_checkpoint(self, ckpt_path: str, lr: float | None = None):
        self.log_info(f"Loading checkpoint from {ckpt_path}")

        ckpt_state = torch.load(ckpt_path, map_location=self.map_location)
        self.epochs_metrics.load_state_dict(ckpt_state["metrics"]["steps"])
        self.validation_metrics.load_state_dict(ckpt_state["metrics"]["validation"])
        self.module.load_state_dict(ckpt_state["module"], lr=lr)
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.current_step = ckpt_state["step"]
        self.current_epoch = ckpt_state["epoch"] + 1
        # n_train_batches = self.datamodule.total_batches["train"]
        # limit_batches = (
        #     self._limit_batches if self._limit_batches > 0 else n_train_batches
        # )
        # epochs_run = (self.current_step + 1) // limit_batches
        # if epochs_run == self.current_epoch + 1:
        #     self.current_epoch += 1
        self.log_info(
            f"The training is resumed at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )

    @property
    def device_info(self) -> str:
        return f"[{self.device}] "

    def save_checkpoint(self, ckpt_path: str):
        self.log_info(f"Saving checkpoint to {ckpt_path}")
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
        self.log_info(
            f"The training is saved at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )
