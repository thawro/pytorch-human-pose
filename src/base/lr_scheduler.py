from typing import Literal

from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler


class LRScheduler:
    def __init__(self, lr_scheduler: TorchLRScheduler, interval: Literal["epoch", "step"]):
        self.lr_scheduler = lr_scheduler
        self.interval = interval

    def step(self):
        self.lr_scheduler.step()

    def state_dict(self) -> dict:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return str(
            {
                "interval": self.interval,
                "lr_scheduler": self.lr_scheduler.__class__.__name__,
                "last_epoch": self.lr_scheduler.last_epoch,
            }
        )

    @property
    def is_step_interval(self) -> bool:
        return self.interval == "step"

    @property
    def is_epoch_interval(self) -> bool:
        return self.interval == "epoch"
