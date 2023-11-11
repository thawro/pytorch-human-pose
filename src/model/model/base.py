from torch import Tensor, nn
from torchinfo import summary
from abc import abstractmethod
import torch


class BaseModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
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
    def example_input(self, device: str = "cpu") -> dict[str, Tensor]:
        raise NotImplementedError()

    def export_to_onnx(self, filename: str = "model.onnx"):
        dynamic_axes = {
            name: {0: "batch_size"} for name in self.input_names + self.output_names
        }
        torch.onnx.export(
            self,
            self.example_input(),
            filename,
            export_params=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=dynamic_axes,
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
