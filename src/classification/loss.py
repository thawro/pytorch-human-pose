from torch import nn, Tensor
from src.base.loss import BaseLoss, WeightedLoss
from torch.nn.modules.loss import _Loss
import torch


class ClassificationLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, targets: Tensor, logits: Tensor) -> Tensor:
        return self.criterion(logits, targets)
