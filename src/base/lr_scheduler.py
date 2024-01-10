from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler
from typing import Literal


class LRScheduler:
    def __init__(
        self, lr_scheduler: TorchLRScheduler, interval: Literal["epoch", "step"]
    ):
        self.lr_scheduler = lr_scheduler
        self.interval = interval

    def step(self):
        self.lr_scheduler.step()

    def state_dict(self) -> dict:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)

    @property
    def is_step_interval(self) -> bool:
        return self.interval == "step"

    @property
    def is_epoch_interval(self) -> bool:
        return self.interval == "epoch"
