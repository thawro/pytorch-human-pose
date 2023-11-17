from torch import Tensor, nn
from src.base.model import BaseImageModel


class KeypointsModel(BaseImageModel):
    net: nn.Module

    def __init__(self, net: nn.Module):
        input_names = ["images"]
        output_names = ["keypoints"]
        super().__init__(net, input_names, output_names)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 256, 256)

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)

    def detect_keypoints(self, images: Tensor) -> Tensor:
        keypoints = self.net(images)
        return keypoints
