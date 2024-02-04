from ..trainer import Trainer
from ..module import BaseModule
from ..datamodule import DataModule
from src.utils.model import seed_everything
from typing import Callable


from torch.distributed import init_process_group, destroy_process_group
import torch.backends.cudnn as cudnn

import torch
import os


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train(
    train_fn: Callable,
    use_distributed: bool = True,
    use_fp16: bool = True,
    seed: int = 42,
):
    if use_distributed:
        ddp_setup()
    if use_fp16:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."
    seed_everything(seed)
    train_fn()
    destroy_process_group()
