from torch import Tensor, nn
from src.base.model import BaseImageModel
from src.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


class BaseKeypointsModel(BaseImageModel):
    def __init__(self, net: nn.Module, num_keypoints: int):
        super().__init__(net, ["images"], ["keypoints"])
        self.num_keypoints = num_keypoints

    def init_weights(
        self, ckpt_path: str | None, map_location: dict, verbose: bool = False
    ):
        super().init_weights(ckpt_path, map_location, verbose)
        log.info("=> init weights from normal distribution [Keypoints]")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)


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
