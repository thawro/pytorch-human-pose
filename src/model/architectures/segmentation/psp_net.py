"""Implementation of PSPNet (with dilated ResNet)
https://arxiv.org/pdf/1612.01105.pdf

Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import model_zoo

from src.model.architectures.helpers import ConvBnAct
from .base import SegmentationNet

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: tuple[int, ...] | list[int] = (3, 4, 23, 3),
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3


def load_state_dict(model: ResNet, state_dict: dict):
    state_dict.pop("fc.weight")
    state_dict.pop("fc.bias")
    model.load_state_dict(state_dict)


def resnet(version: str, pretrained: bool = True) -> ResNet:
    if version == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2])
    elif version == "resnet34":
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    elif version == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3])
    elif version == "resnet101":
        model = ResNet(Bottleneck, [3, 4, 23, 3])
    elif version == "resnet152":
        model = ResNet(Bottleneck, [3, 8, 36, 3])
    else:
        raise ValueError("Wrong model version")

    if pretrained:
        load_state_dict(model, model_zoo.load_url(model_urls[version]))
    return model


class PoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: int | tuple[int]):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.channel_dim_red = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[2:]  # x is of shape NCHW
        pooled_feature_map = self.pool(x)
        dim_reduced_feature_map = self.channel_dim_red(pooled_feature_map)
        upsampled_feature = F.interpolate(dim_reduced_feature_map, size=input_size, mode="bilinear")
        return upsampled_feature


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        pool_in_channels: int,
        pool_out_channels: int,
        bins: tuple[int, ...],
    ):
        super().__init__()
        self.bins = bins
        pooling_modules = [
            PoolingModule(pool_in_channels, pool_out_channels, pool_size) for pool_size in bins
        ]
        self.ppm = nn.ModuleList(pooling_modules)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        pooled_features = [x] + [pooling_module(x) for pooling_module in self.ppm]
        return self.relu(torch.cat(pooled_features, dim=1))


BACKBONE_3_4_CHANNELS = {
    "resnet18": [256, 512],
    "resnet34": [256, 512],
    "resnet50": [1024, 2048],
    "resnet101": [1024, 2048],
    "resnet152": [1024, 2048],
}


class PSPNet(SegmentationNet):
    def __init__(
        self,
        num_classes: int,
        bins: tuple[int, ...] = (1, 2, 3, 6),
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet101",
        cls_dropout: float = 0.5,
    ):
        super().__init__(num_classes)
        encoder = resnet(backbone, pretrained=True)
        self.init_conv_0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.encoder_1 = encoder.layer1
        self.encoder_2 = encoder.layer2
        self.encoder_3 = encoder.layer3
        self.encoder_4 = encoder.layer4

        encoder_3_channels, encoder_4_channels = BACKBONE_3_4_CHANNELS[backbone]

        pool_out_channels = encoder_4_channels // 4
        self.ppm = PyramidPoolingModule(encoder_4_channels, pool_out_channels, bins)
        ppm_out_channels = encoder_4_channels + len(bins) * pool_out_channels

        self.pre_head_conv = ConvBnAct(ppm_out_channels, 512, 1)  # check kernel 3
        self.dropout = nn.Dropout2d(p=0.2)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8),
        )

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(cls_dropout),
            nn.Linear(encoder_3_channels, num_classes),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.init_conv_0(x)
        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out_3 = self.encoder_3(out)  # out_3
        out = self.encoder_4(out_3)

        out = self.ppm(out)
        out = self.pre_head_conv(out)
        out = self.dropout(out)
        seg_out = self.segmentation_head(out)
        cls_out = self.classification_head(out_3)
        return seg_out, cls_out
