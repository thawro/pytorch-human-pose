from torch import Tensor, nn
from src.base.model import BaseImageModel


class BaseClassificationModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["logits"])


class ClassificationModel(BaseClassificationModel):
    def forward(self, images: Tensor) -> list[Tensor]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 224, 224)
