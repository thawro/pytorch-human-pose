from ..trainer import Trainer
from ..module import BaseModule
from ..datamodule import DataModule
from src.utils.model import seed_everything


from torch.distributed import init_process_group, destroy_process_group
import torch.backends.cudnn as cudnn

import torch
import os


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train(
    trainer: Trainer,
    module: BaseModule,
    datamodule: DataModule,
    pretrained_ckpt_path: str | None,
    ckpt_path: str | None,
    seed: int = 42,
):
    seed_everything(seed)
    if trainer.use_distributed:
        ddp_setup()
    if trainer.use_fp16:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."
    trainer.fit(
        module,
        datamodule,
        pretrained_ckpt_path=pretrained_ckpt_path,
        ckpt_path=ckpt_path,
    )
    destroy_process_group()
