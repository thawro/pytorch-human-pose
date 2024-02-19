import torch
from typing import Literal
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.logger.pylogger import logged_tqdm, log
from src.logger.loggers import Loggers, Status
import torch.distributed as dist

from .datamodule import DataModule
from .module import BaseModule
from .callbacks import BaseCallback, Callbacks
from .meters import Meters
from .storage import MetricsStorage
import random
import os
import sys
import logging

_stage = Literal["train", "val", "eval_val"]
_accelerator = Literal["cpu", "gpu"]


# TODO: add loggers (MLFlow) logging of params, metrics and .log files


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: Loggers,
        accelerator: _accelerator,
        callbacks: list[BaseCallback],
        file_log: logging.Logger, 
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
        self.file_log = file_log
        self.accelerator = accelerator
        self._set_device()
        self.callbacks = Callbacks(callbacks, device_id=self.device_id)
        self.logger = logger
        # self.logger.start_run(self.device_info)
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
            self.logger.log_metrics(optizers_lr, self.current_epoch)
            storage.append(
                optizers_lr, self.current_step, self.current_epoch, split="train"
            )
        for stage in stages:
            metrics = self.meters[stage].to_dict()
            storage.append(metrics, self.current_step, self.current_epoch, stage)
            stage_metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
            self.logger.log_metrics(stage_metrics, self.current_epoch)
            

    def evaluate(self, dataloader: DataLoader, stage: _stage):
        """Evaluate on validation set"""
        self.module.model.net.eval()
        meters = self.meters[stage]
        meters.reset()
        limit_batches = self.get_limit_batches(dataloader)
        random_idx = random.randint(0, limit_batches - 1)
        
        def fn(batch, batch_idx: int, random_idx: int, meters: Meters, limit_batches: int, stage: str, trainer: "Trainer"):
            trainer.batch_to_device(batch)
            val_metrics, val_results = trainer.module.validation_step(
                batch, batch_idx, stage=stage
            )
            meters.update(val_metrics, batch[0].shape[0])

            if stage == "eval_val" and batch_idx == random_idx:
                self.results.extend(val_results)
            trainer.callbacks.on_step_end(self)
            batch_idx += 1
            limit_batches -= 1
            is_break = limit_batches == -1
            kwargs = dict(batch_idx=batch_idx, random_idx=random_idx, meters=meters, limit_batches=limit_batches, stage=stage, trainer=trainer)
            return kwargs, is_break

        with torch.no_grad():
            kwargs = dict(batch_idx=0, random_idx=random_idx, meters=meters, limit_batches=limit_batches, stage=stage, trainer=self)
            tqdm_iter = tqdm(dataloader, leave=True, desc=stage, ncols=100, total=limit_batches)
            logged_tqdm(self.file_log, tqdm_iter, fn, kwargs)
            meters.all_reduce()

    def sanity_check(self, dataloader: DataLoader):
        """Run sanity check"""
        self.module.model.net.eval()
        limit_batches = 20
        
        def fn(batch, batch_idx: int, limit_batches: int, stage: str, trainer: "Trainer"):
            trainer.batch_to_device(batch)
            trainer.module.validation_step(
                batch, batch_idx, stage=stage
            )
            trainer.callbacks.on_step_end(self)
            limit_batches -= 1
            is_break = limit_batches == -1
            kwargs = dict(batch_idx=batch_idx, limit_batches=limit_batches, stage=stage, trainer=trainer)
            return kwargs, is_break
        
        with torch.no_grad():
            kwargs = dict(batch_idx=0, limit_batches=limit_batches, stage="sanity_check", trainer=self)
            tqdm_iter = tqdm(dataloader, leave=True, desc="Sanity check", ncols=100, total=limit_batches)
            logged_tqdm(self.file_log, tqdm_iter, fn, kwargs)

    def single_epoch(self, train_dataloader: DataLoader):
        self.module.model.net.train()
        meters = self.meters["train"]
        meters.reset()
        limit_batches = self.get_limit_batches(train_dataloader)
        
        def fn(batch, batch_idx: int, limit_batches: int, meters: Meters, trainer: "Trainer"):
            trainer.batch_to_device(batch)
            train_metrics = trainer.module.training_step(
                batch, batch_idx
            )
            meters.update(train_metrics, batch[0].shape[0])
            trainer.callbacks.on_step_end(self)
            trainer.current_step += 1
            trainer.module.set_attributes(current_step=trainer.current_step)
            batch_idx += 1
            limit_batches -= 1
            is_break = limit_batches == -1
            kwargs = dict(batch_idx=batch_idx, limit_batches=limit_batches, meters=meters, trainer=trainer)
            return kwargs, is_break
        
        kwargs = dict(batch_idx=0, limit_batches=limit_batches, meters=meters, trainer=self)
        tqdm_iter = tqdm(train_dataloader, leave=True, desc="Train", ncols=100, total=limit_batches)
        logged_tqdm(self.file_log, tqdm_iter, fn, kwargs)            
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
        start_epoch = int(self.current_epoch)
        end_epoch = start_epoch + self.max_epochs # TODO: should end_epoch be max_epochs or start+max?
        try:
            for epoch in range(
                start_epoch, end_epoch
            ):
                if self.use_distributed:
                    train_dataloader.sampler.set_epoch(epoch)  # DDP
                self.current_epoch = epoch
                self.module.on_epoch_start()
                self.callbacks.on_epoch_start(self)
                module.set_attributes(current_epoch=epoch)
                self.single_epoch(train_dataloader)
                self.evaluate(val_dataloader, stage="val")
                self._update_metrics(stages=["train", "val"])
                self.module.on_epoch_end()
                self.callbacks.on_epoch_end(self)
                self.results.clear()
                self.log_info(f" <<<  epoch {epoch} finished  >>> ")
                if self.use_distributed:
                    dist.barrier()
        except KeyboardInterrupt as e:
            self.log_error(str(e) + "KeyboardInterrupt")
            self.callbacks.on_failure(self, Status.KILLED)
            self.logger.finalize(Status.KILLED)
            raise e
        self.logger.finalize(status=Status.FINISHED)
    
        
    def log_info(self, msg: str) -> None:
        log.info(f"{self.device_info}{msg}")
        
    def log_warn(self, msg: str) -> None:
        log.warn(f"{self.device_info}{msg}")
        
    def log_error(self, msg: str) -> None:
        log.error(f"{self.device_info}{msg}")
    
    def log_exception(self, exception: Exception) -> None:
        log.exception(exception)

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
        logger_state = self.logger.state_dict()

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "metrics": metrics_state,
            "callbacks": callbacks_state,
            "logger": logger_state,
            "epoch": self.current_epoch,
            "step": self.current_step,
        }
        torch.save(ckpt_state, ckpt_path)
        self.log_info(
            f"The training is saved at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )
