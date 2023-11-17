from torch import nn
from src.base.loss import BaseLoss, WeightedLoss


class DummyLoss(BaseLoss):
    def __init__(self):
        loss_criterion = nn.MSELoss()
        loss_fn = WeightedLoss(loss_criterion, weight=1)
        super().__init__(loss_fn=loss_fn)
