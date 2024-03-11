"""Other utilities"""

import os
import random
from contextlib import contextmanager
from datetime import datetime
from timeit import default_timer

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def random_float(min: float, max: float) -> float:
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


def get_device_and_id(accelerator: str, use_DDP: bool) -> tuple[str, int]:
    if accelerator == "gpu" and torch.cuda.is_available():
        if use_DDP and "LOCAL_RANK" in os.environ:
            device_id = int(os.environ["LOCAL_RANK"])
        else:
            device_id = 0
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
        device_id = 0
    return device, device_id


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start
