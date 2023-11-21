from torch import nn, Tensor
from src.base.architectures.helpers import ConvBnAct


class ResidualModule(nn.Module):
    expansion: int = 2

    def __init__(self, in_channels: int, mid_channels: int) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv_layers = nn.Sequential(
            ConvBnAct(in_channels, mid_channels, 1),
            ConvBnAct(mid_channels, mid_channels, 3),
            ConvBnAct(mid_channels, out_channels, 1, activation=None),
        )
        if out_channels == in_channels:
            self.conv_residual = nn.Identity()
        else:
            self.conv_residual = ConvBnAct(
                in_channels, out_channels, kernel_size=1, activation=None
            )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv_residual(x)
        out = self.conv_layers(x)
        return self.relu(out + residual)


class HourglassModule(nn.Module):
    def __init__(
        self, num_blocks: int = 4, in_channels: int = 256, mid_channels: int = 128
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        layers_down = []
        layers_residual = []
        layers_up = []
        for i in range(num_blocks):
            block_down = nn.Sequential(
                nn.MaxPool2d(2, 2), ResidualModule(in_channels, mid_channels)
            )

            block_residual = ResidualModule(in_channels, mid_channels)

            block_up = nn.Sequential(
                ResidualModule(in_channels, mid_channels),
                nn.UpsamplingNearest2d(scale_factor=2),
            )

            layers_down.append(block_down)
            layers_residual.append(block_residual)
            layers_up.append(block_up)

            in_channels = mid_channels * ResidualModule.expansion

        self.layers_down = nn.ModuleList(layers_down)
        self.layers_residual = nn.ModuleList(layers_residual)
        self.mid_conv = ResidualModule(in_channels, mid_channels)
        self.layers_up = nn.ModuleList(layers_up)

    def forward(self, x: Tensor) -> Tensor:
        residuals = []
        for i in range(self.num_blocks):
            residual = self.layers_residual[i](x)
            residuals.append(residual)
            x = self.layers_down[i](x)
        x = self.mid_conv(x)
        for i in range(self.num_blocks):
            residual = residuals[-(i + 1)]
            x = self.layers_up[i](x)
            x += residual
        return x


class HourglassHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_keypoints: int):
        super().__init__()
        self.conv_0 = nn.Sequential(
            ResidualModule(in_channels, mid_channels),
            ConvBnAct(in_channels, in_channels, 1),
        )

        self.heatmaps_head = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.remap_feats = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.remap_heatmaps = nn.Conv2d(num_keypoints, in_channels, kernel_size=1)

    def forward(self, hg_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hg_feats = self.conv_0(hg_out)
        heatmaps = self.heatmaps_head(hg_feats)
        remaped_heatmaps = self.remap_heatmaps(heatmaps)
        remaped_feats = self.remap_feats(hg_feats)
        return remaped_feats, heatmaps, remaped_heatmaps


class HourglassNet(nn.Module):
    def __init__(self, num_stages: int, num_keypoints: int = 17) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.stem = ConvBnAct(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.layer_0 = ResidualModule(in_channels=64, mid_channels=64)
        self.maxpool_0 = nn.MaxPool2d(2, 2)
        self.layer_1 = ResidualModule(in_channels=128, mid_channels=128)
        in_channels = 256
        mid_channels = 128
        self.layer_2 = ResidualModule(in_channels, mid_channels)
        stages = []
        heatmap_heads = []
        for i in range(num_stages):
            stage = HourglassModule(4, in_channels, mid_channels)
            heatmap_head = HourglassHead(in_channels, mid_channels, num_keypoints)
            stages.append(stage)
            heatmap_heads.append(heatmap_head)

        self.stages = nn.ModuleList(stages)
        self.heatmap_heads = nn.ModuleList(heatmap_heads)

    def forward(self, x: Tensor) -> list[Tensor]:
        out = self.stem(x)
        out = self.layer_0(out)
        out = self.maxpool_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)

        stages_heatmaps = []
        for i in range(self.num_stages):
            residual = out
            hg_out = self.stages[i](out)
            after_hg_feats, heatmaps, heatmaps_feats = self.heatmap_heads[i](hg_out)
            # heatmaps = nn.functional.sigmoid(heatmaps)
            stages_heatmaps.append(heatmaps)
            out = residual + after_hg_feats + heatmaps_feats
        return stages_heatmaps
