from src.keypoints.architectures.hrnet import HRNetBackbone, Bottleneck
from torch import nn, Tensor
import torch
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, C: int, num_classes: int = 1000):
        super().__init__()
        stages_C = [C, 2 * C, 4 * C, 8 * C]
        out_channels = [128, 256, 512, 1024]
        chann_incr_blocks = []
        self.num_stages = len(stages_C)
        for i in range(self.num_stages):
            block = Bottleneck(in_channels=stages_C[i], out_channels=out_channels[i])
            chann_incr_blocks.append(block)
        self.chann_incr_blocks = nn.ModuleList(chann_incr_blocks)

        downsample_blocks = []
        for i in range(self.num_stages - 1):
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels[i + 1]),
                nn.ReLU(inplace=True),
            )
            downsample_blocks.append(block)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x_stages: list[Tensor]) -> Tensor:
        out = self.chann_incr_blocks[0](x_stages[0])
        for i in range(self.num_stages - 1):
            downsampled = self.downsample_blocks[i](out)
            out = self.chann_incr_blocks[i + 1](x_stages[i + 1]) + downsampled
        out = self.final_conv(out)

        if torch._C._get_tracing_state():
            flat = out.flatten(start_dim=2).mean(dim=2)
        else:
            flat = F.avg_pool2d(out, kernel_size=out.size()[2:]).view(out.size(0), -1)

        logits = self.classifier(flat)
        return logits


class ClassificationHRNet(nn.Module):
    def __init__(self, C: int = 32, num_classes: int = 1000):
        self.stages_C = [C, 2 * C, 4 * C, 8 * C]
        super().__init__()
        self.backbone = HRNetBackbone(C)
        self.classification_head = ClassificationHead(C, num_classes=num_classes)

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        high_res_out = self.backbone(images)
        logits = self.classification_head(high_res_out)
        return logits


if __name__ == "__main__":
    from src.utils.model import get_model_summary
    from torchinfo import summary

    net = ClassificationHRNet(C=32)
    x = torch.randn((1, 3, 224, 224))
    out = net(x)
    print(out.shape)

    txt = get_model_summary(net, [x])
    print(txt)
    # summary(net, input_data=x, col_names=["num_params"], depth=10)
