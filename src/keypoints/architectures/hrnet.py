from torch import nn, Tensor
from src.base.architectures.helpers import ConvBnAct
from typing import Type
import torch


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = int(out_channels / self.expansion)
        self.conv_layers = nn.Sequential(
            ConvBnAct(in_channels, mid_channels, 1),
            ConvBnAct(mid_channels, mid_channels, 3),
            ConvBnAct(mid_channels, out_channels, 1, activation=None),
        )
        self.conv_residual = nn.Identity()
        if out_channels != in_channels:
            self.conv_residual = ConvBnAct(
                in_channels, out_channels, kernel_size=1, activation=None
            )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv_residual(x)
        out = self.conv_layers(x)
        return self.relu(out + residual)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = int(out_channels / self.expansion)
        self.conv_layers = nn.Sequential(
            ConvBnAct(in_channels, mid_channels, 3),
            ConvBnAct(mid_channels, out_channels, 3, activation=None),
        )
        self.conv_residual = nn.Identity()
        if out_channels != in_channels:
            self.conv_residual = ConvBnAct(
                in_channels, out_channels, kernel_size=1, activation=None
            )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv_residual(x)
        out = self.conv_layers(x)
        return self.relu(out + residual)


class HighResolutionBlock(nn.Module):
    _idx: int = 0

    def __init__(
        self,
        num_units: int,
        ResidUnitType: Type[BasicBlock | BottleneckBlock],
        num_scales: int,
        num_in_channels: list[int],
        num_out_channels: list[int],
        is_final_block: bool,
    ):
        super().__init__()
        HighResolutionBlock._idx += 1
        self.idx = int(HighResolutionBlock._idx)
        self.num_units = num_units
        self.is_final_block = is_final_block
        self.num_scales = num_scales
        self.num_scales_out = num_scales
        if is_final_block:
            self.num_scales_out += 1

        scales_blocks = []
        for i in range(num_scales):
            single_scale_block_units = []
            in_channels = num_in_channels[i]
            out_channels = num_out_channels[i]
            for j in range(num_units):
                residual_unit = ResidUnitType(in_channels, out_channels)
                in_channels = out_channels
                single_scale_block_units.append(residual_unit)
            scale_block_units = nn.Sequential(*single_scale_block_units)
            scales_blocks.append(scale_block_units)
        self.scales_blocks = nn.ModuleList(scales_blocks)

        # fusion operators
        self.apply_fusion = num_scales > 0  # TODO
        if self.apply_fusion:
            scales_fusion_layers = []
            # i - index of output scale block
            # j - index of input scale block
            for i in range(self.num_scales_out):
                scale_fusion_layers = []
                for j in range(self.num_scales):
                    num_high2low = i - j if i > j else 0
                    num_mid2mid = 1 if i == j else 0
                    num_low2high = j - i if j > i else 0
                    if num_high2low > 0:
                        layer = nn.Sequential(
                            *[
                                ConvBnAct(
                                    num_out_channels[k + j],
                                    num_out_channels[k + j + 1],
                                    3,
                                    2,
                                    True,
                                    None,
                                )
                                for k in range(num_high2low)
                            ]
                        )
                    elif num_mid2mid > 0:
                        layer = nn.Identity()
                    elif num_low2high > 0:
                        layer = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2**num_low2high),
                            ConvBnAct(
                                num_out_channels[j],
                                num_out_channels[i],
                                1,
                                1,
                                activation=None,
                            ),
                        )
                    else:
                        raise ValueError("Errur")
                    scale_fusion_layers.append(layer)
                scale_fusion_layers = nn.ModuleList(scale_fusion_layers)
                scales_fusion_layers.append(scale_fusion_layers)
            self.scales_fusion_layers = nn.ModuleList(scales_fusion_layers)

    def forward(self, scales_inputs: list[Tensor]):
        print("Block: ", self.idx, " (final)" if self.is_final_block else "")
        # forward pass in each scale units
        for scale_in in scales_inputs:
            print(scale_in.shape, end="")
        print()
        scales_out = []
        for i in range(self.num_scales):
            scale_block = self.scales_blocks[i]
            scale_input = scales_inputs[i]
            scale_out = scale_block(scale_input)
            scales_out.append(scale_out)

        # multiresolution fusion after each block
        if self.apply_fusion:
            fusion_scales_out = []
            # i - index of output scale block
            # j - index of input scale block
            for i in range(self.num_scales_out):
                scale_out = 0
                for j in range(self.num_scales):
                    fusion_layer = self.scales_fusion_layers[i][j]
                    scale_fusion_layer_input = scales_out[j]
                    print(i, j)
                    print(fusion_layer, scale_fusion_layer_input.shape)
                    fusion_layer_out = fusion_layer(scale_fusion_layer_input)
                    print(fusion_layer_out.shape)
                    scale_out += fusion_layer_out
                    print()

                fusion_scales_out.append(scale_out)
            scales_out = fusion_scales_out
        for scale_in in scales_out:
            print(scale_in.shape, end="")
        print()
        print()
        return scales_out


class HighResolutionStage(nn.Module):
    _idx: int = 0

    def __init__(
        self,
        num_blocks: int,
        num_units: int,
        ResidUnitType: Type[BasicBlock | BottleneckBlock],
        num_scales: int,
        num_in_channels: list[int],
        num_out_channels: list[int],
        is_final_stage: bool,
    ):
        super().__init__()
        HighResolutionStage._idx += 1
        self.idx = int(HighResolutionStage._idx)
        self.is_final_stage = is_final_stage
        blocks = []
        for i in range(num_blocks):
            is_final_block = (i == num_blocks - 1) and not is_final_stage
            block = HighResolutionBlock(
                num_units=num_units,
                ResidUnitType=ResidUnitType,
                num_scales=num_scales,
                num_in_channels=num_in_channels,
                num_out_channels=num_out_channels,
                is_final_block=is_final_block,
            )
            for j in range(len(num_in_channels)):
                num_in_channels[j] = num_out_channels[j]
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, scales_inputs: list[Tensor]):
        print("Stage: ", self.idx)
        scales_outs = self.blocks(scales_inputs)
        return scales_outs


class HRNet(nn.Module):
    def __init__(self):
        super().__init__()
        C = 32
        config = [
            # num_blocks, num_units, ResidUnitType, num_scales, num_in_channels, num_out_channels
            [1, 4, BottleneckBlock, 1, [64], [256, 2 * C]],
            [1, 4, BasicBlock, 2, [256, 2 * C], [C, 2 * C, 4 * C]],
            [4, 4, BasicBlock, 3, [C, 2 * C, 4 * C], [C, 2 * C, 4 * C, 8 * C]],
            [3, 4, BasicBlock, 4, [C, 2 * C, 4 * C, 8 * C], [C, 2 * C, 4 * C, 8 * C]],
        ]
        self.conv1 = ConvBnAct(3, 64, 3, 2)
        self.conv2 = ConvBnAct(64, 64, 3, 2)

        stages = []
        for i, stage_cfg in enumerate(config):
            blocks, units, ResidUnitType, scales, in_chans, out_chans = stage_cfg
            is_final_stage = i == len(config) - 1
            stage = HighResolutionStage(
                blocks,
                units,
                ResidUnitType,
                scales,
                in_chans,
                out_chans,
                is_final_stage,
            )
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.stages([x])
        return out


if __name__ == "__main__":
    net = HRNet()

    x = torch.randn(16, 3, 256, 256)

    out = net(x)
