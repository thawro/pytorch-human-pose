"""Train the model"""

import torch
from src.logging import TerminalLogger, get_pylogger

from src.model.module import Trainer
from src.model.utils import seed_everything

from src.bin.utils import create_datamodule, create_callbacks, create_module
from src.bin.config import cfg

log = get_pylogger(__name__)


def main() -> None:
    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule(cfg)
    module = create_module(cfg)

    logger = TerminalLogger(cfg.logs_path, config=cfg.to_dict())
    callbacks = create_callbacks(cfg)

    trainer = Trainer(
        logger=logger,
        device=cfg.setup.device,
        callbacks=callbacks,
        max_epochs=cfg.setup.max_epochs,
        limit_batches=cfg.setup.limit_batches,
        log_every_n_steps=cfg.setup.log_every_n_steps,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
