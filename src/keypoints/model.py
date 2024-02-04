from torch import Tensor, nn
from src.base.model import BaseImageModel


class BaseKeypointsModel(BaseImageModel):
    def __init__(self, net: nn.Module, num_keypoints: int):
        super().__init__(net, ["images"], ["keypoints"])
        self.num_keypoints = num_keypoints


class KeypointsModel(BaseKeypointsModel):
    def forward(self, images: Tensor) -> list[Tensor]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 256, 256)


class AEKeypointsModel(BaseKeypointsModel):
    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 512, 512)
