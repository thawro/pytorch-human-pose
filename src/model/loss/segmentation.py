from torch import Tensor
from torch.nn.modules.loss import _Loss
from .base import WeightedLoss


class AuxiliarySegmentationLoss(_Loss):
    def __init__(self, seg_loss: WeightedLoss, cls_loss: WeightedLoss):
        super().__init__()
        self.seg_loss = seg_loss
        self.cls_loss = cls_loss

    def calculate_loss(
        self, seg_pred: Tensor, seg_target: Tensor, cls_pred: Tensor, cls_target: Tensor
    ) -> Tensor:
        seg_loss = self.seg_loss(seg_pred, seg_target)
        cls_loss = self.seg_loss(cls_pred, cls_target)
        return seg_loss + cls_loss
