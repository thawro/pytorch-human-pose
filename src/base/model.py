from torch import Tensor, nn
from torchinfo import summary
from abc import abstractmethod
import torch


class BaseModel(nn.Module):
    def __init__(
        self,
        net: nn.DataParallel,
        input_names: list[str] = ["input"],
        output_names: list[str] = ["output"],
    ):
        super().__init__()
        self.net = net
        self.input_names = input_names
        self.output_names = output_names

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def example_input(self) -> dict[str, Tensor]:
        raise NotImplementedError()

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            name: {0: "batch_size"} for name in self.input_names + self.output_names
        }

    def export_to_onnx(self, filepath: str = "model.onnx"):
        torch.onnx.export(
            self.net.module,
            self.example_input(),
            filepath,
            export_params=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes(),
        )

    def summary(self, depth: int = 4):
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        return str(
            summary(
                self,
                input_data=self.example_input(),
                depth=depth,
                col_names=col_names,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def export_layers_description_to_txt(self, filepath: str) -> str:
        return str(self.net)


class BaseImageModel(BaseModel):
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        input_axes = {
            name: {0: "batch_size", 2: "height", 3: "width"}
            for name in self.input_names
        }
        output_axes = {name: {0: "batch_size"} for name in self.output_names}
        return input_axes | output_axes

    def example_input(
        self,
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 256,
        width: int = 256,
    ) -> dict[str, Tensor]:
        return {
            "images": torch.randn(
                batch_size, num_channels, height, width, device=self.device
            )
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)
