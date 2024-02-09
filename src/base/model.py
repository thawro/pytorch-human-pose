from torch import Tensor, nn
from torchinfo import summary
from abc import abstractmethod
import torch
from src.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


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

    def forward(self, x: Tensor):
        return self.net(x)

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
            self,
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

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def export_layers_description_to_txt(self, filepath: str) -> str:
        return str(self)

    def init_pretrained_weights(
        self, ckpt_path: str | None, map_location: dict, verbose: bool = False
    ):
        if ckpt_path is None:
            return
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)
        log.info(f"=> loading pretrained model {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)

        state_dict = {}
        for name, m in ckpt.items():
            if name in parameters_names or name in buffers_names:
                if verbose:
                    log.info(f"=> init {name} from {ckpt_path}")
                state_dict[name] = m
        self.load_state_dict(state_dict, strict=False)

    def init_weights(
        self, ckpt_path: str | None, map_location: dict, verbose: bool = False
    ):
        self.init_pretrained_weights(ckpt_path, map_location, verbose)


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
