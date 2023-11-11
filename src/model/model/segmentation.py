from torch import Tensor
from .base import BaseModel
from src.model.architectures.segmentation.base import SegmentationNet
import torch


class SegmentationModel(BaseModel):
    net: SegmentationNet

    def example_input(self, batch_size: int = 1) -> dict[str, Tensor]:
        return {"images": torch.randn(batch_size, 3, 256, 256, device=self.device)}

    def segment(self, images: Tensor) -> Tensor:
        seg_out, cls_out = self.net(images)
        return seg_out
