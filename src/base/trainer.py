import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.logger.loggers import Loggers, Status
from src.logger.pylogger import log, log_breaking_point, logged_tqdm
from src.utils.types import _accelerator, _stage
from src.utils.utils import get_device_and_id

from .callbacks import BaseCallback, Callbacks
from .datamodule import DataModule
from .meters import Meters
from .module import BaseModule
from .storage import MetricsStorage, SystemMonitoringStorage

# TODO:
# fix compile + DDP

# Gradients visualization (with plotly), ideas:
# https://gist.github.com/Flova/8bed128b41a74142a661883af9e51490
# https://github.com/wandb/wandb/blob/main/wandb/wandb_torch.py#L121
# https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/histogram.py

# test classification pipeline
# gifs saving from training evaluation samples (must ensure that sample_idxs are same)


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: Loggers,
        accelerator: _accelerator,
        callbacks: list[BaseCallback],
        max_epochs: int = 100,
        limit_batches: int = -1,
        use_DDP: bool = False,
        sync_batchnorm: bool = True,
        use_compile: bool = False,
        run_sanity_check: bool = False,
    ):
        stages = ["train", "val", "eval_val"]
        self.use_DDP = use_DDP
        self.sync_batchnorm = sync_batchnorm
        self.use_compile = use_compile
        self.run_sanity_check = run_sanity_check
        self.meters = {stage: Meters(use_DDP=use_DDP) for stage in stages}
        self.accelerator = accelerator
        self.device, self.device_id = get_device_and_id(self.accelerator, self.use_DDP)
        self.callbacks = Callbacks(callbacks, device_id=self.device_id)
        self.logger = logger
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.current_step = -1
        self.current_epoch = 0
        self.epochs_metrics = MetricsStorage(name="Epochs")  # every step metrics
        self.validation_metrics = MetricsStorage(name="LogStep")  # validation metrics
        self.system_monitoring = SystemMonitoringStorage()
        self.results = []

    @property
    def map_location(self) -> dict[str, str]:
        return {"cuda:0": self.device}

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
            metrics = {
                f"{name}_LR": optimizer.param_groups[0]["lr"]
                for name, optimizer in self.module.optimizers.items()
            }
            metrics["epoch"] = self.current_epoch
            metrics["step"] = self.current_step
            self.logger.log_metrics(metrics, self.current_epoch)
            storage.append(metrics, self.current_step, self.current_epoch, split="train")
        for stage in stages:
            metrics = self.meters[stage].to_dict()
            storage.append(metrics, self.current_step, self.current_epoch, stage)
            stage_metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
            self.logger.log_metrics(stage_metrics, self.current_epoch)

    def evaluate(self, dataloader: DataLoader, stage: _stage, sanity: bool = False):
        """Evaluate on validation set"""
        self.module.model.net.eval()

        meters = self.meters[stage]
        meters.reset()
        limit_batches = self.get_limit_batches(dataloader)

        random_idx = random.randint(0, limit_batches - 1)

        if sanity:
            limit_batches = 10
            random_idx = 0

        def fn(
            batch,
            batch_idx: int,
            random_idx: int,
            meters: Meters,
            limit_batches: int,
            stage: str,
            trainer: "Trainer",
        ):
            is_break = limit_batches == 0
            if is_break:
                return {}, is_break
            val_metrics, val_results = trainer.module._validation_step(
                batch, batch_idx, stage=stage
            )
            meters.update(val_metrics, batch[0].shape[0])

            if batch_idx == random_idx and not sanity:
                self.results.extend(val_results)
            trainer.callbacks.on_step_end(self)
            batch_idx += 1
            limit_batches -= 1
            kwargs = dict(
                batch_idx=batch_idx,
                random_idx=random_idx,
                meters=meters,
                limit_batches=limit_batches,
                stage=stage,
                trainer=trainer,
            )
            return kwargs, is_break

        with torch.no_grad():
            kwargs = dict(
                batch_idx=0,
                random_idx=random_idx,
                meters=meters,
                limit_batches=limit_batches,
                stage=stage,
                trainer=self,
            )
            tqdm_iter = tqdm(dataloader, leave=True, desc=stage, ncols=100, total=limit_batches)
            logged_tqdm(self.logger.file_log, tqdm_iter, fn, kwargs)
            meters.all_reduce()

    def sanity_check(self, dataloader: DataLoader):
        """Run sanity check"""
        self.evaluate(dataloader, stage="sanity", sanity=True)

    def single_epoch(self, train_dataloader: DataLoader):
        self.module.model.net.train()
        meters = self.meters["train"]
        meters.reset()
        limit_batches = self.get_limit_batches(train_dataloader)

        def fn(
            batch,
            batch_idx: int,
            limit_batches: int,
            meters: Meters,
            trainer: "Trainer",
        ):
            is_break = limit_batches == 0
            if is_break:
                return {}, is_break
            train_metrics = trainer.module._training_step(batch, batch_idx)
            meters.update(train_metrics, batch[0].shape[0])
            trainer.callbacks.on_step_end(self)
            trainer.current_step += 1
            trainer.module.set_attributes(current_step=trainer.current_step)
            batch_idx += 1
            limit_batches -= 1
            kwargs = dict(
                batch_idx=batch_idx,
                limit_batches=limit_batches,
                meters=meters,
                trainer=trainer,
            )
            return kwargs, is_break

        kwargs = dict(batch_idx=0, limit_batches=limit_batches, meters=meters, trainer=self)
        tqdm_iter = tqdm(train_dataloader, leave=True, desc="Train", ncols=100, total=limit_batches)
        logged_tqdm(self.logger.file_log, tqdm_iter, fn, kwargs)
        meters.all_reduce()

    def _wait_for_all_workers(self):
        if self.use_DDP:
            dist.barrier()

    def on_epoch_start(self):
        pass

    def fit(
        self,
        module: BaseModule,
        datamodule: DataModule,
        pretrained_ckpt_path: str | None = None,
        ckpt_path: str | None = None,
    ):
        train_dataloader = datamodule.train_dataloader
        val_dataloader = datamodule.val_dataloader

        log.info(
            "Dataloaders utilisation (per GPU):\n"
            f"  Train: {self.get_limit_batches(train_dataloader)}/{len(train_dataloader)} batches\n"
            f"  Val  : {self.get_limit_batches(val_dataloader)}/{len(val_dataloader)}  batches"
        )

        module.pass_attributes(
            device_id=self.device_id,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            datamodule=datamodule,
            limit_batches=int(self._limit_batches),
        )
        # Correct order (?):
        # Compile -> CUDA -> weights init -> pretrained weights load -> previous run ckpt load -> DDP

        if self.use_compile and not self.use_DDP:
            module.model.compile()
        if self.accelerator == "gpu":
            module.model.to_CUDA(self.device_id)
            module.loss_fn.cuda(self.device_id)

        module.model.init_weights()
        if pretrained_ckpt_path is None:
            log.warn("..Skipping pretrained weights loading (pretrained_ckpt_path is None)..")
        else:
            log.info(f"..Loading pretrained checkpoint (from '{pretrained_ckpt_path}')..")
            pretrained_ckpt = torch.load(pretrained_ckpt_path, map_location=self.map_location)
            # NOTE: its trainer ckpt, so we need to extract model state from it
            if "module" in pretrained_ckpt:
                pretrained_ckpt = pretrained_ckpt["module"]["model"]
            module.model.init_pretrained_weights(pretrained_ckpt)

        self.module = module
        self.datamodule = datamodule
        self.module.set_optimizers()
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        if self.use_DDP:
            self.module.model.to_DDP(self.device_id, self.sync_batchnorm)
            if self.use_compile:
                self.module.model.compile()

        self.module.set_attributes(current_step=self.current_step)

        self.callbacks.on_fit_start(self)

        if self.run_sanity_check:
            self.sanity_check(val_dataloader)

        self._wait_for_all_workers()
        if ckpt_path is None:
            msg = "<<<  Training started  >>>"
        else:
            msg = f"<<<  The training is resumed at: epoch={self.current_epoch}, step={self.current_step}  >>>"
        log_breaking_point(
            msg, n_top=2, n_bottom=2, top_char="*", bottom_char="*", num_chars=100, worker=0
        )
        start_epoch = int(self.current_epoch)
        try:
            for epoch in range(start_epoch, self.max_epochs):
                log_breaking_point(
                    f"<<<  Epoch {epoch} started  >>>", n_top=1, top_char=" ", worker=0
                )
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)  # train_dl uses shuffle=True
                self.current_epoch = epoch
                self.on_epoch_start()
                self.module.on_epoch_start()
                self.callbacks.on_epoch_start(self)
                self.module.set_attributes(current_epoch=epoch)
                self.single_epoch(train_dataloader)
                self.evaluate(val_dataloader, stage="val")
                self._update_metrics(stages=["train", "val"])
                self._wait_for_all_workers()
                self.module.on_epoch_end()
                self.callbacks.on_epoch_end(self)
                self._wait_for_all_workers()
                self.results.clear()
                log_breaking_point(
                    f"<<<  Epoch {epoch} finished  >>>", n_bottom=1, worker=0, bottom_char="="
                )
        except KeyboardInterrupt as e:
            log.error(str(e) + "KeyboardInterrupt")
            self.callbacks.on_failure(self, Status.KILLED)
            self.logger.finalize(Status.KILLED)
            raise e
        self.logger.finalize(status=Status.FINISHED)

    def load_checkpoint(self, ckpt_path: str):
        log.info(f"..Loading checkpoint from '{ckpt_path}'..")
        ckpt_state = torch.load(ckpt_path, map_location=self.map_location)
        self.epochs_metrics.load_state_dict(ckpt_state["metrics"]["steps"])
        self.validation_metrics.load_state_dict(ckpt_state["metrics"]["validation"])
        self.module.load_state_dict(ckpt_state["module"])
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        step, epoch = ckpt_state["step"], ckpt_state["epoch"]
        self.current_step = step
        self.current_epoch = epoch + 1
        log.info(f"     Loaded current_step ({step}) and current_epoch ({epoch}) state")

    def save_checkpoint(self, ckpt_path: str):
        if self.device_id != 0:  # save only for cuda:0 (DDP)
            return
        log.info(
            f"Saving checkpoint (epoch={self.current_epoch}, step={self.current_step}) to '{ckpt_path}'"
        )
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
