import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
from src.utils.model import save_checkpoint

from src.base.datamodule import DataModule
from src.base.module import BaseModule
from src.base.callbacks import BaseCallback, Callbacks

log = get_pylogger(__name__)


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: BaseLogger,
        device: torch.device | str,
        callbacks: list[BaseCallback],
        max_epochs: int = 100,
        limit_batches: int = -1,
        log_every_n_steps: int = -1,
    ):
        self.logger = logger
        self.device = device
        self.callbacks = Callbacks(callbacks)
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.log_every_n_steps = log_every_n_steps
        self.current_step = 0
        self.current_epoch = 0
        self.best_metrics = {}

    def batch_to_device(self, batch) -> None:
        for j in range(len(batch)):
            if isinstance(batch[j], Tensor):
                batch[j] = batch[j].to(self.device)
            elif isinstance(batch[j][0], Tensor):  # list of tensors
                batch[j] = [batch[j][i].to(self.device) for i in range(len(batch[j]))]

    def get_limit_batches(self) -> int:
        if self.current_step == 0:
            return 1
        else:
            return int(self._limit_batches)

    def evaluate(self, dataloader: DataLoader, stage: str):
        """Evaluate on validation set"""
        with torch.no_grad():
            loop = tqdm(dataloader, leave=True, desc=stage)
            limit_batches = self.get_limit_batches()
            for i, batch in enumerate(loop):
                self.batch_to_device(batch)
                self.module.validation_step(batch, i, stage=stage)
                limit_batches -= 1
                if limit_batches == 0:
                    break

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
        self.callbacks.on_epoch_start(self)
        loop = tqdm(self.datamodule.train_dataloader, leave=True, desc="Train")
        limit_batches = int(self._limit_batches)
        for i, batch in enumerate(loop):
            self.batch_to_device(batch)
            self.module.training_step(batch, i)
            self.module.log_optimizer_params()
            self.current_step += 1
            self.module.set_attributes(current_step=self.current_step)
            if self.current_step % self.log_every_n_steps == 0:
                self.callbacks.on_validation_start(self)
                self.evaluate(self.datamodule.val_dataloader, stage="eval_val")
                self.evaluate(self.datamodule.train_dataloader, stage="eval_train")
                self.callbacks.on_validation_end(self)
            limit_batches -= 1
            if limit_batches == 0:
                break
        self.module.on_epoch_end()
        self.callbacks.on_epoch_end(self)

    def fit(self, module: BaseModule, datamodule: DataModule):
        n_train_batches = datamodule.total_batches["train"] - 1
        limit_batches = (
            self._limit_batches if self._limit_batches > 0 else n_train_batches
        )
        if self.log_every_n_steps < 0:
            self.log_every_n_steps = abs(self.log_every_n_steps) * min(
                n_train_batches, limit_batches
            )
        self.datamodule = datamodule
        module.model = module.model.to(self.device)
        self.module = module
        self.module.pass_attributes(
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            datamodule=datamodule,
            limit_batches=int(self._limit_batches),
            log_every_n_steps=self.log_every_n_steps,
        )
        self.callbacks.on_fit_start(self)
        self.module.set_attributes(current_step=self.current_step)

        self.sanity_check(self.datamodule.val_dataloader)
        try:
            for epoch in range(
                self.current_epoch, self.current_epoch + self.max_epochs
            ):
                self.current_epoch = epoch
                module.set_attributes(current_epoch=epoch)
                self.single_epoch()
                self.callbacks.on_epoch_end(self)
                print()
        except KeyboardInterrupt as e:
            self.callbacks.on_failure(self)
            raise e

    def load_checkpoint(self, ckpt_path: str, lr: float | None = None):
        log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_state = torch.load(ckpt_path)
        self.module.load_state_dict(ckpt_state["module"], lr=lr)
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.current_step = ckpt_state["step"]
        self.current_epoch = math.ceil(
            self.current_step / self.datamodule.total_batches["train"]
        )
        log.info(
            f"The training is resumed at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )

    def save_checkpoint(self, ckpt_path: str):
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        callbacks_state = self.callbacks.state_dict()
        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "callbacks": callbacks_state,
            "epoch": self.current_epoch,
            "step": self.current_step,
        }
        save_checkpoint(ckpt_state, ckpt_path)
        log.info(
            f"The training is saved at: "
            f"epoch={self.current_epoch}, "
            f"step={self.current_step}"
        )
