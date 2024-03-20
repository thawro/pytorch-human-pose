import torch
from torch import Tensor, nn

from src.keypoints.architectures.hrnet import BasicBlock, HRNetBackbone


class DeconvHeatmapsHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kpts: int,
        num_resid_blocks: int = 4,
        kernel_size: int = 4,
        padding: int = 1,
        output_padding: int = 0,
    ):
        super().__init__()
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
        self.final_layer = nn.Conv2d(out_channels, num_kpts, 1, 1, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.deconv(x)
        feats = self.resid_blocks(out)
        heatmaps = self.final_layer(feats)
        return feats, heatmaps


class HigherHRNet(nn.Module):
    def __init__(self, num_kpts: int, C: int = 32):
        super().__init__()
        self.backbone = HRNetBackbone(C, final_stage_single_scale=True)
        self.num_kpts = num_kpts
        self.init_heatmaps_head = nn.Conv2d(C, num_kpts * 2, 1, 1, 0)

        deconv_channels = [C + num_kpts * 2, C]
        self.num_deconv_layers = len(deconv_channels) - 1

        deconv_layers = []
        for i in range(self.num_deconv_layers):
            in_channels = deconv_channels[i]
            out_channels = deconv_channels[i + 1]
            deconv_layer = DeconvHeatmapsHead(in_channels, out_channels, num_kpts)
            deconv_layers.append(deconv_layer)

        self.deconv_layers = nn.ModuleList(deconv_layers)

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        high_res_out = self.backbone(images)[0]
        feats = high_res_out
        init_heatmaps = self.init_heatmaps_head(high_res_out)
        out = init_heatmaps
        heatmaps = [init_heatmaps]
        for i in range(self.num_deconv_layers):
            deconv_input = torch.cat((feats, out), 1)
            deconv_layer = self.deconv_layers[i]
            feats, out = deconv_layer(deconv_input)
            heatmaps.append(out)

        tags_heatmaps = init_heatmaps[:, self.num_kpts :]
        stages_kpts_heatmaps = [hm[:, : self.num_kpts] for hm in heatmaps]

        return stages_kpts_heatmaps, tags_heatmaps


if __name__ == "__main__":
    from thop import profile
    from torchinfo import summary

    net = HigherHRNet(num_kpts=17)

    x = torch.randn(1, 3, 224, 224)

    ops, params, layer_dict = profile(net, inputs=(x,), ret_layer_info=True)
    print(ops, params)
    print(type(ops), type(params))

    col_names = ["input_size", "output_size", "num_params", "params_percent"]
    summary_txt = str(summary(net, input_data=x, depth=4, col_names=col_names))
