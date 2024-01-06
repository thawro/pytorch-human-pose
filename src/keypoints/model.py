from torch import Tensor, nn
from src.base.model import BaseImageModel
from abc import abstractmethod


class BaseKeypointsModel(BaseImageModel):
    net: nn.Module

    def __init__(self, net: nn.Module, num_keypoints: int):
        input_names = ["images"]
        output_names = ["keypoints"]
        self.num_keypoints = num_keypoints
        super().__init__(net, input_names, output_names)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 256, 256)

    @abstractmethod
    def forward(self, images: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def detect_keypoints(self, images: Tensor):
        raise NotImplementedError()


class KeypointsModel(BaseKeypointsModel):
    net: nn.Module

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)

    def detect_keypoints(self, images: Tensor) -> Tensor:
        keypoints = self(images)
        return keypoints


class AEKeypointsModel(BaseKeypointsModel):
    net: nn.Module

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.net(images)

    def detect_keypoints(self, images: Tensor) -> Tensor:
        keypoints, tags = self(images)
        # TODO
        return keypoints
