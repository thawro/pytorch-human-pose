"""Implementation of PSPNet (with dilated ResNet)
https://arxiv.org/pdf/1612.01105.pdf

Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
"""

from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.architectures.helpers import ConvBnAct
from .base import SegmentationNet
from src.model.architectures.backbones.resnet import resnet


class PoolingModule(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, pool_size: int | tuple[int]
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.channel_dim_red = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[2:]  # x is of shape NCHW
        pooled_feature_map = self.pool(x)
        dim_reduced_feature_map = self.channel_dim_red(pooled_feature_map)
        upsampled_feature = F.interpolate(
            dim_reduced_feature_map, size=input_size, mode="bilinear"
        )
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
            PoolingModule(pool_in_channels, pool_out_channels, pool_size)
            for pool_size in bins
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
        self.init_conv_0 = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool
        )
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
