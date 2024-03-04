import os

import torch
import torch.backends.cudnn as cudnn
from torch.distributed import destroy_process_group, init_process_group

from src.logger.loggers import Status
from src.logger.pylogger import log
from src.utils.model import seed_everything
from src.utils.utils import get_rank

from ..config import BaseConfig


def ddp_setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_id = int(os.environ["LOCAL_RANK"])
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device_id)


def ddp_finalize():
    log.info("..Destroying process group..")
    destroy_process_group()


def train(cfg: BaseConfig):
    if cfg.trainer.use_DDP:
        ddp_setup()
    log.info(f"..Current device: {torch.cuda.current_device()}..")
    log.info(f"..Starting {cfg.device} process..")
    rank = get_rank()
    log.info(
        f"..Increasing seed from {cfg.setup.seed} to {cfg.setup.seed + rank} for worker {rank}.."
    )
    cfg.setup.seed += rank
    seed_everything(cfg.setup.seed)

    datamodule = cfg.create_datamodule()
    module = cfg.create_module()
    trainer = cfg.create_trainer()

    try:
        cudnn.benchmark = cfg.cudnn.benchmark
        cudnn.deterministic = cfg.cudnn.deterministic
        cudnn.enabled = cfg.cudnn.enabled
        trainer.fit(
            module,
            datamodule,
            pretrained_ckpt_path=cfg.setup.pretrained_ckpt_path,
            ckpt_path=cfg.setup.ckpt_path,
        )
        ddp_finalize()
    except Exception as e:
        log.exception(e)
        trainer.callbacks.on_failure(trainer, Status.FAILED)
        trainer.logger.finalize(Status.FAILED)
        ddp_finalize()
        raise e
