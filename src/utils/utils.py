"""Other utilities"""

import random
from datetime import datetime
import os
import torch

def random_float(min: float, max: float):
    return random.random() * (max - min) + min


def get_current_date_and_time() -> str:
    now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H:%M")
    dt_string = now.strftime("%m-%d_%H:%M")
    return dt_string

def prepend_exception_message(exception: Exception, prefix: str):
    _args = exception.args
    if len(_args) >= 1:
        exception.args = (f"{prefix}{_args[0]}", *_args[1:])


def get_device_and_id(accelerator: str, use_distributed: bool) -> tuple[str, int]:
    if accelerator == "gpu" and torch.cuda.is_available():
        if use_distributed and "LOCAL_RANK" in os.environ:
            device_id = int(os.environ["LOCAL_RANK"])
        else:
            device_id = 0
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
        device_id = 0
    return device, device_id