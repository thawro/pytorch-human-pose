"""
The paper implementation differs a bit from what is on the main repo. The main notes:
* Fusion is done between blocks, so in total 1 + 4 + 3 = 8 fusions are done
* After the last fusion in the stage there is a new branch created
    * It isnt created by aggregated result of stride-2 convolutions from higher branches
    * It is created by applying stride-2 conv3x3 on the lowest scale
* There is a transition layer between stages, that works differently for each stage connection:
    * Between stage1 and stage2 there is a conv transition layer (for same scales) which is used to
        transform 256 channels to C channels
    * Between stage2-stage3 and stage3-stage4 there are no transition layers (for all scales)

Notes about my implementation:
I have separate classes for:
    * residual units (Bottleneck | BasicBlock)
    * high resolution blocks (HighResolutionBlock) - composed of residual units only, 
        passes inputs in separate scale branches
    * fusion layers (FusionLayer) - used after each hr-block. It uses  stride-2-conv / identity / conv+upsampl to 
        map features to desired scale/depth and fuses those features with sum operation
    * transition layers (TransitionLayer) - used after fusion layer of last block of a stage. It maps 
        256 channels to C channels in the stage1-stage2 transition and creates new scale branches for all transitions
"""

from torch import nn, Tensor
from src.base.architectures.helpers import ConvBnAct
from typing import Type
import torch


class Bottleneck(nn.Module):
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

    def __init__(self, in_channels: int, **kwargs) -> None:
        super().__init__()
        out_channels = in_channels * self.expansion
        self.conv_layers = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 3),
            ConvBnAct(in_channels, out_channels, 3, activation=None),
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
    def __init__(
        self,
        num_units: int,
        ResidUnitType: Type[BasicBlock | Bottleneck],
        num_in_channels: list[int],
    ):
        super().__init__()
        self.num_units = num_units
        self.num_scales = len(num_in_channels)

        num_out_channels = []

        scales_blocks = []
        for i in range(self.num_scales):
            single_scale_block_units = []
            in_channels = num_in_channels[i]
            out_channels = in_channels * ResidUnitType.expansion
            for j in range(num_units):
                residual_unit = ResidUnitType(in_channels, out_channels=out_channels)
                in_channels = out_channels
                single_scale_block_units.append(residual_unit)
            num_out_channels.append(out_channels)
            scale_block_units = nn.Sequential(*single_scale_block_units)
            scales_blocks.append(scale_block_units)
        self.scales_blocks = nn.ModuleList(scales_blocks)
        self.num_out_channels = num_out_channels

    def forward(self, scales_inputs: list[Tensor]) -> list[Tensor]:
        # forward pass in each scale units
        scales_out = []
        for i in range(self.num_scales):
            scale_block = self.scales_blocks[i]
            scale_input = scales_inputs[i]
            scale_out = scale_block(scale_input)
            scales_out.append(scale_out)
        return scales_out


class FusionLayer(nn.Module):
    def __init__(self, num_out_channels: list[int], num_scales_out: int = -1):
        super().__init__()
        self.num_scales = len(num_out_channels)
        if num_scales_out == -1:
            num_scales_out = self.num_scales
        self.num_scales_out = num_scales_out
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
                    # high to low is matching channels only in last high2low block
                    high2low_blocks = []
                    for k in range(num_high2low):
                        is_last = k == num_high2low - 1
                        in_chans = num_out_channels[j]
                        out_chans = num_out_channels[j + k + 1] if is_last else in_chans
                        high2low_block = ConvBnAct(
                            in_chans, out_chans, 3, 2, True, None
                        )
                        high2low_blocks.append(high2low_block)
                    layer = nn.Sequential(*high2low_blocks)
                elif num_mid2mid > 0:
                    layer = nn.Identity()
                elif num_low2high > 0:
                    layer = nn.Sequential(
                        ConvBnAct(
                            num_out_channels[j],
                            num_out_channels[i],
                            1,
                            1,
                            activation=None,
                        ),
                        nn.UpsamplingNearest2d(scale_factor=2**num_low2high),
                    )
                else:
                    raise ValueError("Errur")
                scale_fusion_layers.append(layer)
            scale_fusion_layers = nn.ModuleList(scale_fusion_layers)
            scales_fusion_layers.append(scale_fusion_layers)
        self.scales_fusion_layers = nn.ModuleList(scales_fusion_layers)

    def forward(self, scales_outputs: list[Tensor]) -> list[Tensor]:
        fusion_scales_out = []
        # i - index of output scale block
        # j - index of input scale block
        for i in range(self.num_scales_out):
            scale_out = 0
            for j in range(self.num_scales):
                fusion_layer = self.scales_fusion_layers[i][j]
                scale_fusion_layer_input = scales_outputs[j]
                fusion_layer_out = fusion_layer(scale_fusion_layer_input)
                scale_out += fusion_layer_out

            fusion_scales_out.append(scale_out)
        return fusion_scales_out


class TransitionLayer(nn.Module):
    def __init__(
        self,
        num_in_channels: list[int],
        num_out_channels: list[int],
        is_first_stage: bool,
    ):
        super().__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.is_first_stage = is_first_stage
        layers = []
        # transition block for each scale branch
        for i in range(len(num_in_channels)):
            in_channels, out_channels = num_in_channels[i], num_out_channels[i]
            # not clear in the paper if scaleX - scaleX transition is identity or
            # convolution, the implementation from github uses:
            # conv for transition between stage1 and stage2,
            # identity for stage2-stage3 and stage4-stage4
            if is_first_stage:
                transition_block = ConvBnAct(in_channels, out_channels, 3, 1)
            else:
                transition_block = nn.Identity()
            layers.append(transition_block)

        # additional transition block for the new branch
        in_channels, out_channels = num_in_channels[-1], num_out_channels[-1]
        transition_block = ConvBnAct(in_channels, out_channels, 3, 2)
        layers.append(transition_block)
        self.transition_blocks = nn.ModuleList(layers)

    def forward(self, stage_scales_outputs: list[Tensor]) -> list[Tensor]:
        transition_out = []
        for i in range(len(self.num_in_channels)):
            stage_scale_out = stage_scales_outputs[i]
            transition_block = self.transition_blocks[i]
            transition_scale_out = transition_block(stage_scale_out)
            transition_out.append(transition_scale_out)

        last_stage_scale_out = stage_scales_outputs[-1]
        last_transition_block = self.transition_blocks[-1]
        new_scale_out = last_transition_block(last_stage_scale_out)
        transition_out.append(new_scale_out)
        return transition_out


class HighResolutionStage(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_units: int,
        ResidUnitType: Type[BasicBlock | Bottleneck],
        num_in_channels: list[int],
        num_out_channels: list[int],
        is_final_stage: bool,
        is_first_stage: bool,
    ):
        # num_out_channels is bigger than num_in_channels between stages because
        # we need to use the last element of num_out_channels to create the new scale branch
        # using transition layer
        super().__init__()
        self.is_final_stage = is_final_stage
        blocks = []
        for i in range(num_blocks):
            is_final_block = i == num_blocks - 1

            highres_block = HighResolutionBlock(
                num_units=num_units,
                ResidUnitType=ResidUnitType,
                num_in_channels=num_in_channels,
            )
            _num_out_channels = num_out_channels[: len(num_in_channels)]
            if is_final_stage and is_final_block:
                num_scales_out = 1
            else:
                num_scales_out = len(_num_out_channels)
            fusion_layer = FusionLayer(_num_out_channels, num_scales_out)
            block = [highres_block, fusion_layer]
            for j in range(len(num_in_channels)):
                num_in_channels[j] = num_out_channels[j]
            blocks.extend(block)
        self.blocks = nn.Sequential(*blocks)

        # transition layer uses last high resolution blocks out channels as in channels
        transition_in_channels = highres_block.num_out_channels
        if not is_final_stage:
            self.transition_layer = TransitionLayer(
                transition_in_channels, num_out_channels, is_first_stage
            )
        else:
            self.transition_layer = None

    def forward(self, scales_inputs: list[Tensor]) -> list[Tensor]:
        scales_outs = self.blocks(scales_inputs)
        if self.transition_layer is not None:
            scales_outs = self.transition_layer(scales_outs)
        return scales_outs


class HRNet(nn.Module):
    def __init__(self, num_keypoints: int, C: int = 32):
        super().__init__()
        self.num_keypoints = num_keypoints
        C_2, C_4, C_8 = 2 * C, 4 * C, 8 * C
        config = [
            # num_blocks, num_units, ResidUnitType, num_scales, num_in_channels, num_out_channels
            [1, 4, Bottleneck, [64], [C, C_2]],
            [1, 4, BasicBlock, [C, C_2], [C, C_2, C_4]],
            [4, 4, BasicBlock, [C, C_2, C_4], [C, C_2, C_4, C_8]],
            [3, 4, BasicBlock, [C, C_2, C_4, C_8], [C, C_2, C_4, C_8]],
        ]
        self.conv1 = ConvBnAct(3, 64, 3, 2)
        self.conv2 = ConvBnAct(64, 64, 3, 2)

        stages = []
        for i, stage_cfg in enumerate(config):
            blocks, units, ResidUnitType, in_chans, out_chans = stage_cfg
            is_final = i == len(config) - 1
            is_first = i == 0
            stage = HighResolutionStage(
                blocks, units, ResidUnitType, in_chans, out_chans, is_final, is_first
            )
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.final_conv = nn.Conv2d(C, num_keypoints, 1, 1, 0)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.stages([x])
        high_res_out = out[0]
        heatmaps = self.final_conv(high_res_out)
        return [heatmaps]


if __name__ == "__main__":
    net = HRNet(num_keypoints=17)

    x = torch.randn(16, 3, 256, 256)

    out = net(x)
    print(out.shape)
