from src.base.architectures.backbones.resnet import ResNet
from torch import nn, Tensor
import torch


class SimpleBaseline(nn.Module):
    def __init__(self, num_keypoints: int = 17):
        super().__init__()
        self.num_keypoints = num_keypoints
        backbone: ResNet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        num_deconv_layers = 3
        deconv_kernel = 4
        deconv_padding = 1
        decovn_output_padding = 0

        deconv_channels = [512, 256, 256, 256]

        deconv_layers = []
        for i in range(num_deconv_layers):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=deconv_channels[i],
                    out_channels=deconv_channels[i + 1],
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=deconv_padding,
                    output_padding=decovn_output_padding,
                    bias=False,
                )
            )

        self.deconv_layers = nn.Sequential(*deconv_layers)

        self.final_layer = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
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
