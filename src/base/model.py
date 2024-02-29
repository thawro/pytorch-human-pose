from abc import abstractmethod

import torch
from torch import Tensor, nn
from torchinfo import summary

from src.logger.pylogger import log


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
        return {name: {0: "batch_size"} for name in self.input_names + self.output_names}

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

    def parse_checkpoint(self, ckpt: dict) -> dict:
        # TODO: dont know why DDP saves the names as _orig_mod
        redundant_prefixes = ["module.", "_orig_mod."]
        for key in list(ckpt.keys()):
            renamed_key = str(key)
            for prefix in redundant_prefixes:
                renamed_key = renamed_key.replace(prefix, "")
            ckpt[renamed_key] = ckpt.pop(key)
        return ckpt

    def init_pretrained_weights(self, ckpt_path: str, map_location: dict):
        log.info(f"..Loading pretrained weights (from {ckpt_path})..")
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        # NOTE: its trainer ckpt, so we need to extract model state from it
        ckpt = ckpt["module"]["model"]

        total_from = len(ckpt.keys())  # numer of params in checkpoints module
        total_to = len(parameters_names) + len(buffers_names)  # number of params in current module
        total_loaded = 0

        ckpt = self.parse_checkpoint(ckpt)

        state_dict = {}
        for name, m in ckpt.items():
            if name in parameters_names or name in buffers_names:
                total_loaded += 1
                state_dict[name] = m
        log_method = log.warn if total_loaded == 0 else log.info
        log_method(
            f"\tLoaded {total_loaded} parameters (total_from = {total_from}, total_to = {total_to})"
        )
        self.load_state_dict(state_dict, strict=False)

    @abstractmethod
    def init_weights(self):
        raise NotImplementedError()


class BaseImageModel(BaseModel):
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        input_axes = {name: {0: "batch_size", 2: "height", 3: "width"} for name in self.input_names}
        output_axes = {name: {0: "batch_size"} for name in self.output_names}
        return input_axes | output_axes

    def example_input(
        self,
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 256,
        width: int = 256,
    ) -> dict[str, Tensor]:
        return {"images": torch.randn(batch_size, num_channels, height, width, device=self.device)}
