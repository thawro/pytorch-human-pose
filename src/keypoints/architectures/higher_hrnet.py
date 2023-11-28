from .hrnet import HRNet, BasicBlock
from torch import nn, Tensor
import torch


class DeconvHeatmapsHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_keypoints: int,
        num_resid_blocks: int = 4,
        kernel_size: int = 4,
        padding: int = 1,
        output_padding: int = 0,
    ):
        stride = 2
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        resid_blocks = []
        for i in range(num_resid_blocks):
            resid_blocks.append(BasicBlock(out_channels))
        self.resid_blocks = nn.Sequential(*resid_blocks)
        self.final_layer = nn.Conv2d(out_channels, num_keypoints, 1, 1, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.deconv(x)
        feats = self.resid_blocks(x)
        heatmaps = self.final_layer(feats)
        return feats, heatmaps


class HigherHRNet(nn.Module):
    def __init__(self, num_keypoints: int, C: int = 32):
        super().__init__()
        hrnet = HRNet(num_keypoints, C)
        self.stem = nn.Sequential(
            hrnet.conv1,
            hrnet.conv2,
        )
        self.hrnet_backbone = hrnet.stages

        self.tags_head = nn.Conv2d(C, num_keypoints, 1, 1, 0)
        self.heatmaps_head = nn.Conv2d(C, num_keypoints, 1, 1, 0)

        deconv_channels = [C + num_keypoints, C]
        self.num_deconv_layers = len(deconv_channels) - 1

        deconv_layers = []
        for i in range(self.num_deconv_layers):
            in_channels = deconv_channels[i]
            out_channels = deconv_channels[i + 1]
            deconv_layer = DeconvHeatmapsHead(in_channels, out_channels, num_keypoints)
            deconv_layers.append(deconv_layer)

        self.deconv_layers = nn.ModuleList(deconv_layers)

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        x = self.stem(x)
        high_res_out = self.hrnet_backbone([x])[0]
        tags = self.tags_head(high_res_out)

        feats = high_res_out
        out = tags
        heatmaps = []
        for i in range(self.num_deconv_layers):
            deconv_input = torch.cat((feats, out), 1)
            deconv_layer = self.deconv_layers[i]
            feats, out = deconv_layer(deconv_input)
            heatmaps.append(out)
        return [tags], heatmaps
