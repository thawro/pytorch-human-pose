"""Implementation of loss Base and weighted loss classes"""

from torch import Tensor
from torch.nn.modules.loss import _Loss
from abc import abstractmethod


class WeightedLoss(_Loss):
    def __init__(self, criterion: _Loss, weight: float):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, *args, **kwargs) -> Tensor:
        loss = self.criterion(*args, **kwargs)
        return self.weight * loss


class BaseLoss(_Loss):
    def __init__(self, loss_fn: WeightedLoss):
        super().__init__()
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()
