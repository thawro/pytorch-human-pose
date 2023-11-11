"""Utility functions for model operations (checkpoint save/load, onnx export, etc.)"""

from typing import Any
import random
import numpy as np
import torch
from torch import Tensor, nn
from torchinfo import summary

from src.logging import get_pylogger

log = get_pylogger(__name__)


def save_checkpoint(ckpt: dict[str, dict[str, Any]], path: str = "ckpt.pt") -> None:
    log.info(f"Saving checkpoint to {path}")
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    log.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def export_to_onnx(
    net: nn.Module,
    dummy_input: Tensor,
    filepath: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    log.info(f"Exporting onnx model to {filepath}")
    torch.onnx.export(
        net,
        dummy_input,
        filepath,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )


def export_to_txt(net: nn.Module, filepath: str) -> str:
    log.info(f"Saving model txt to {filepath}")
    modules_txt = str(net)
    with open(filepath, "w") as text_file:
        text_file.write(modules_txt)
    return modules_txt


def export_summary_to_txt(
    net: nn.Module, dummy_input: Tensor, filepath: str, device: torch.device
) -> str:
    log.info(f"Saving model summary to {filepath}")
    model_summary = str(
        summary(
            net,
            input_data=dummy_input,
            depth=10,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
            ],
            verbose=0,
            device=device,
        )
    )
    with open(filepath, "w") as text_file:
        text_file.write(model_summary)
    return model_summary


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
