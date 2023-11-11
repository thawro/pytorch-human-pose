import torch
from tqdm.auto import tqdm

from src.data.datamodule import DataModule
from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
from src.model.utils import save_checkpoint
from src.model.module.base import BaseModule
from src.callbacks import BaseCallback, Callbacks

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

    def evaluate(
        self,
        dataloader,
        desc: str,
        update_metrics: bool = False,
        single_batch: bool = True,
    ):
        """Evaluate on validation set
        update_metrics=False and single_batch=True for examples plotter callbacks
        update_metrics=True and single_batch=False for full validation set evaluation
        """
        with torch.no_grad():
            loop = tqdm(dataloader, leave=False, desc=desc)
            if single_batch:
                limit_batches = 1
            else:
                limit_batches = int(self._limit_batches)
            for i, batch in enumerate(loop):
                for j in range(len(batch)):
                    batch[j] = batch[j].to(self.device)
                self.module.validation_step(batch, i, update_metrics)
                limit_batches -= 1
                if limit_batches == 0:
                    break

    def train_epoch(self):
        self.module.on_train_epoch_start()
        self.callbacks.on_train_epoch_start(self)
        loop = tqdm(self.datamodule.train_dataloader, leave=False, desc="Train")
        limit_batches = int(self._limit_batches)
        for i, batch in enumerate(loop):
            for j in range(len(batch)):
                batch[j] = batch[j].to(self.device)
            if self.current_step % self.log_every_n_steps == 0:
                self.evaluate(
                    self.datamodule.val_dataloader,
                    desc="Val",
                    update_metrics=False,
                    single_batch=True,
                )
                self.callbacks.log(self)
            self.module.training_step(batch, i, update_metrics=True)
            self.current_step += 1
            self.module.set_attributes(current_step=self.current_step)
            limit_batches -= 1
            if limit_batches == 0:
                break
        self.module.on_train_epoch_end()
        self.callbacks.on_train_epoch_end(self)

    def val_epoch(self):
        self.module.on_validation_epoch_start()
        self.callbacks.on_validation_epoch_start(self)
        self.evaluate(
            self.datamodule.val_dataloader,
            desc="Val",
            update_metrics=True,
            single_batch=False,
        )
        self.module.on_validation_epoch_end()
        self.callbacks.on_validation_epoch_end(self)

    def fit(self, module: BaseModule, datamodule: DataModule):
        if self.log_every_n_steps == -1:
            self.log_every_n_steps = datamodule.total_batches["train"] - 1
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
        self.module.set_attributes(current_step=self.current_step)
        self.callbacks.on_fit_start(self)
        for epoch in range(self.current_epoch, self.current_epoch + self.max_epochs):
            self.current_epoch = epoch
            module.set_attributes(current_epoch=epoch)
            self.train_epoch()
            self.val_epoch()
            module.on_epoch_end()
            self.callbacks.on_epoch_end(self)
            print()

    def load_checkpoint(self, ckpt_path: str):
        log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_state = torch.load(ckpt_path)
        self.module.load_state_dict(ckpt_state["module"])
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.current_epoch = ckpt_state["epoch"] + 1
        self.current_step = ckpt_state["step"] + 1

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
