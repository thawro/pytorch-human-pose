from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from src.base.results import BaseResult
from src.logger.pylogger import log
from src.utils.model import parse_checkpoint


class BaseModel:
    net: nn.Module

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

    def to_CUDA(self, device_id: int):
        log.info(f"..Moving model to CUDA device (cuda:{device_id})..")
        self.net = self.net.cuda(device_id)

    def to_DDP(self, device_id: int, use_batchnorm: bool):
        # NOTE: Issue with BatchNorm for DDP:
        # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/31
        # the forums say that the proper way to use DDP and BatchNorm layers is to use cudnn.enabled = False
        # but it slows the training by 1.5-2x times
        log.info("..Moving Module to DDP (Data Distributed Parallel)..")
        if use_batchnorm:
            log.info("      ..Converting BatchNorm to SyncBatchNorm..")
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = DDP(
            self.net,
            device_ids=[device_id],  # , find_unused_parameters=True
        )

    def compile(self):
        log.info("..Compiling Module (`torch.compile(net)`)..")
        self.net = torch.compile(self.net)

    @property
    def device(self):
        net = self.net.module if isinstance(self.net, DDP) else self.net
        return next(net.parameters()).device

    @abstractmethod
    def example_input(self) -> dict[str, Tensor]:
        raise NotImplementedError()

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {name: {0: "batch_size"} for name in self.input_names + self.output_names}

    def export_to_onnx(self, filepath: str = "model.onnx"):
        torch.onnx.export(
            self.net.module if isinstance(self.net, DDP) else self.net,
            self.example_input(),
            filepath,
            export_params=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes(),
        )

    def summary(self, depth: int = 4) -> str:
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        return summary(
            self.net.module if isinstance(self.net, DDP) else self.net,
            input_data=self.example_input(),
            depth=depth,
            col_names=col_names,
            verbose=0,
        ).__str__()

    def freeze(self) -> None:
        for param in self.net.parameters():
            param.requires_grad = False

    def export_layers_description_to_txt(self, filepath: str) -> str:
        return str(self)

    def state_dict(self) -> dict:
        if isinstance(self.net, DDP):
            return self.net.module.state_dict()
        else:
            return self.net.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.net.load_state_dict(state_dict)
        log.info("     Loaded model state")

    def init_pretrained_weights(self, ckpt: dict):
        log.info("..Setting weights according to pretrained checkpoint..")
        parameters_names = set()
        for name, _ in self.net.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.net.named_buffers():
            buffers_names.add(name)

        total_from = len(ckpt.keys())  # numer of params in checkpoints module
        total_to = len(parameters_names) + len(buffers_names)  # number of params in current module
        total_loaded = 0

        ckpt = parse_checkpoint(ckpt)

        state_dict = {}
        for name, m in ckpt.items():
            if name in parameters_names or name in buffers_names:
                total_loaded += 1
                state_dict[name] = m
        log_method = log.warn if total_loaded == 0 else log.info
        log_method(
            f"      Loaded {total_loaded} parameters (total_from = {total_from}, total_to = {total_to})"
        )
        self.net.load_state_dict(state_dict, strict=False)

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


class BaseInferenceModel:
    model_input_shape: tuple[int, int]

    def __init__(
        self,
        net: nn.Module,
        input_size: int = 512,
        device: str = "cuda:0",
        ckpt_path: str | None = None,
    ):
        self.device = device
        self.input_size = input_size
        self.net = net.to(device)
        self.net.eval()
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "module" in ckpt.keys():
            ckpt = ckpt["module"]["model"]
        ckpt = parse_checkpoint(ckpt)
        self.net.load_state_dict(ckpt)
        log.info(f"Loaded checkpoint from {ckpt_path}")

    def prepare_input(self, image: np.ndarray) -> Tensor | Any:
        raise NotImplementedError()

    def __call__(self, raw_image: np.ndarray, annot: list[dict] | None) -> BaseResult:
        raise NotImplementedError()
