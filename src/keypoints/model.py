from torch import Tensor, nn

from src.base.model import BaseImageModel
from src.logger.pylogger import log


class KeypointsModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["keypoints"])

    def init_weights(self):
        log.info("..Initializing weights from normal distribution (Keypoints)..")
        for m in self.net.modules():
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

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 512, 512)
