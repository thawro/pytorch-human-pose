from ..config import BaseConfig
from src.utils.model import seed_everything
from src.utils.utils import prepend_exception_message
from src.logger.loggers import Status
from torch.distributed import init_process_group, destroy_process_group
import torch.backends.cudnn as cudnn


import torch
import os


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train(cfg: BaseConfig):
    if cfg.trainer.use_distributed:
        ddp_setup()
    cfg.log_info(f"..Starting {cfg.device} process..")
    seed_everything(cfg.setup.seed)

    datamodule = cfg.create_datamodule()
    module = cfg.create_module()
    trainer = cfg.create_trainer()
    
    try:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        if trainer.use_fp16:
            assert (
                torch.backends.cudnn.enabled
            ), "fp16 mode requires cudnn backend to be enabled."
        else:
            torch.set_float32_matmul_precision("high")
        trainer.fit(
            module,
            datamodule,
            pretrained_ckpt_path=cfg.setup.pretrained_ckpt_path,
            ckpt_path=cfg.setup.ckpt_path,
        )
        destroy_process_group()
    except Exception as e:
        prepend_exception_message(e, trainer.device_info)
        trainer.log_exception(e)
        trainer.callbacks.on_failure(trainer, Status.FAILED)
        trainer.logger.finalize(Status.FAILED)
        raise e
