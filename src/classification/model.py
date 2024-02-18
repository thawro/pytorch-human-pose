from torch import Tensor, nn
from src.base.model import BaseImageModel



class BaseClassificationModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["logits"])

    def init_weights(
        self, ckpt_path: str | None, map_location: dict
    ):
        super().init_weights(ckpt_path, map_location)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ClassificationModel(BaseClassificationModel):
    def forward(self, images: Tensor) -> list[Tensor]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 224, 224)
