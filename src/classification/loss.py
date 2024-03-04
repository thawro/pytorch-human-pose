from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


class ClassificationLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, targets: Tensor, logits: Tensor) -> Tensor:
        return self.criterion(logits, targets)
