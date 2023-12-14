from src.base.architectures.backbones.resnet import ResNet
from torch import nn, Tensor
import torch
from typing import Literal

_resnets = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class SimpleBaseline(nn.Module):
    """
    https://arxiv.org/pdf/1804.06208.pdf
    """

    def __init__(self, num_keypoints: int, backbone: _resnets):
        super().__init__()
        self.num_keypoints = num_keypoints
        backbone_net: ResNet = torch.hub.load(
            "pytorch/vision:v0.10.0", backbone, pretrained=True
        )

        self.conv1 = backbone_net.conv1
        self.bn1 = backbone_net.bn1
        self.relu = backbone_net.relu
        self.maxpool = backbone_net.maxpool
        self.layer1 = backbone_net.layer1
        self.layer2 = backbone_net.layer2
        self.layer3 = backbone_net.layer3
        self.layer4 = backbone_net.layer4

        num_deconv_layers = 3
        deconv_kernel = 4
        deconv_padding = 1
        decovn_output_padding = 0

        deconv_channels = [512, 256, 256, 256]

        deconv_layers = []
        for i in range(num_deconv_layers):
            in_channels = deconv_channels[i]
            out_channels = deconv_channels[i + 1]
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_padding,
                        output_padding=decovn_output_padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )

        self.deconv_layers = nn.Sequential(*deconv_layers)

        self.final_layer = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, images: Tensor) -> list[Tensor]:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmaps = [x]
        return heatmaps
