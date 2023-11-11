from torch import nn, Tensor
from abc import abstractmethod


class SegmentationNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def segment(self, x: Tensor) -> Tensor:
        seg_out, cls_out = self(x)
        return seg_out
