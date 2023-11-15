from torch import nn, Tensor
from src.model.architectures.helpers import ConvBnAct


class ResidualModule(nn.Module):
    expansion: int = 2

    def __init__(self, in_channels: int, mid_channels: int) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv_layers = nn.Sequential(
            ConvBnAct(in_channels, mid_channels, 1),
            ConvBnAct(mid_channels, mid_channels, 3),
            ConvBnAct(mid_channels, out_channels, 1),
        )
        if out_channels == in_channels:
            self.conv_residual = nn.Identity()
        else:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv_residual(x)
        out = self.conv_layers(x)
        return out + residual


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
                ResidualModule(in_channels, mid_channels),
                nn.MaxPool2d(2, 2),
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

        self.mid_conv = nn.Sequential(
            ResidualModule(in_channels, mid_channels),
            # ResidualModule(in_channels, mid_channels),
        )
        self.layers_up = nn.ModuleList(layers_up)

    def forward(self, x: Tensor) -> Tensor:
        # print()
        # print("--------")
        # print("Hourglass module ", x.shape)
        residuals = []
        for i in range(self.num_blocks):
            residual = self.layers_residual[i](x)
            residuals.append(residual)
            x = self.layers_down[i](x)
        #     print(f"down {i}, residual: {residual.shape}, x: {x.shape}")
        # print("--")
        x = self.mid_conv(x)
        # print("mid: ", x.shape)
        # print("--")
        for i in range(self.num_blocks):
            residual = residuals[-(i + 1)]
            x = self.layers_up[i](x)
            # print(f"up {i}, residual: {residual.shape}, x: {x.shape}")
            x += residual

        # print("--------")
        return x


class HourglassHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_keypoints: int):
        super().__init__()
        self.conv_0 = ResidualModule(in_channels, mid_channels)
        self.heatmaps_head = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

        self.conv_0_1 = ResidualModule(in_channels, mid_channels)
        self.heatmaps_head_conv = nn.Conv2d(num_keypoints, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        conv_0_feats = self.conv_0(x)
        heatmaps = self.heatmaps_head(conv_0_feats)
        heatmaps_feats = self.heatmaps_head_conv(heatmaps)
        conv_0_1_feats = self.conv_0_1(conv_0_feats)
        return conv_0_1_feats, heatmaps, heatmaps_feats


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
            stages_heatmaps.append(heatmaps)
            out = residual + after_hg_feats + heatmaps_feats
        return stages_heatmaps
