"""Utility functions for model operations (checkpoint save/load, onnx export, etc.)"""

import os
import random
from collections import namedtuple

import numpy as np
import torch
from torch import Tensor, nn
from torchinfo import summary

from src.logger.pylogger import log


def export_to_onnx(
    net: nn.Module,
    dummy_input: Tensor,
    filepath: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    torch.onnx.export(
        net,
        dummy_input,
        filepath,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )


def export_to_txt(net: nn.Module, filepath: str) -> str:
    modules_txt = str(net)
    with open(filepath, "w") as text_file:
        text_file.write(modules_txt)
    return modules_txt


def export_summary_to_txt(
    net: nn.Module, dummy_input: Tensor, filepath: str, device: torch.device
) -> str:
    model_summary = str(
        summary(
            net,
            input_data=dummy_input,
            depth=10,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
            ],
            verbose=0,
            device=device,
        )
    )
    with open(filepath, "w") as text_file:
        text_file.write(model_summary)
    return model_summary


def seed_everything(seed: int):
    log.info(f"..Setting seed to {seed}..")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_summary(
    model: nn.Module,
    input_tensors: list[torch.Tensor],
    item_length: int = 26,
    verbose: bool = False,
):
    summary = []

    ModuleDetails = namedtuple(
        "Layer",
        ["name", "input_size", "output_size", "num_parameters", "multiply_adds"],
    )
    hooks = []
    layer_instances = {}

    def add_hooks(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if (
                class_name.find("Conv") != -1
                or class_name.find("BatchNorm") != -1
                or class_name.find("Linear") != -1
            ):
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size())))
                    * torch.prod(torch.LongTensor(list(output.size())[2:]))
                ).item()
            elif isinstance(module, nn.Linear):
                flops = (
                    torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)
                ).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops,
                )
            )

        if (
            not isinstance(module, nn.ModuleList)
            and not isinstance(module, nn.Sequential)
            and module != model
        ):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ""
    if verbose:
        details = (
            "Model Summary"
            + os.linesep
            + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                " " * (space_len - len("Name")),
                " " * (space_len - len("Input Size")),
                " " * (space_len - len("Output Size")),
                " " * (space_len - len("Parameters")),
                " " * (space_len - len("Multiply Adds (Flops)")),
            )
            + os.linesep
            + "-" * space_len * 5
            + os.linesep
        )

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += (
                "{}{}{}{}{}{}{}{}{}{}".format(
                    layer.name,
                    " " * (space_len - len(layer.name)),
                    layer.input_size,
                    " " * (space_len - len(str(layer.input_size))),
                    layer.output_size,
                    " " * (space_len - len(str(layer.output_size))),
                    layer.num_parameters,
                    " " * (space_len - len(str(layer.num_parameters))),
                    layer.multiply_adds,
                    " " * (space_len - len(str(layer.multiply_adds))),
                )
                + os.linesep
                + "-" * space_len * 5
                + os.linesep
            )

    details += (
        os.linesep
        + "Total Parameters: {:,}".format(params_sum)
        + os.linesep
        + "-" * space_len * 5
        + os.linesep
    )
    details += (
        "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(
            flops_sum / (1024**3)
        )
        + os.linesep
        + "-" * space_len * 5
        + os.linesep
    )
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details
