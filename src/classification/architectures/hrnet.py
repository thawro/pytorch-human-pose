from src.keypoints.architectures.hrnet import HRNet, Bottleneck
from src.base.architectures.helpers import ConvBnAct
from torch import nn, Tensor
import torch


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
            block = ConvBnAct(
                in_channels=out_channels[i],
                out_channels=out_channels[i + 1],
                kernel_size=3,
                stride=2,
                batch_norm=True,
                activation="ReLU",
            )
            downsample_blocks.append(block)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)
        self.final_conv = nn.Sequential(
            ConvBnAct(out_channels[-1], 2048, 1, 1, True, "ReLU"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x_stages: list[Tensor]) -> Tensor:
        out_stages = []
        for i in range(self.num_stages):
            x = x_stages[i]
            incr_block = self.chann_incr_blocks[i]
            out_stages.append(incr_block(x))

        for i in range(self.num_stages - 1):
            out = out_stages[i].clone()
            if i > 0:
                out += downsampled
            downsample_block = self.downsample_blocks[i]
            downsampled = downsample_block(out)
        remapped = self.final_conv(downsampled)
        logits = self.classifier(remapped)
        return logits


class ClassificationHRNet(nn.Module):
    def __init__(self, num_keypoints: int, C: int = 32, num_classes: int = 1000):
        self.stages_C = [C, 2 * C, 4 * C, 8 * C]
        super().__init__()
        hrnet = HRNet(num_keypoints, C)
        self.backbone = nn.Sequential(hrnet.conv1, hrnet.conv2, hrnet.stages)
        self.classification_head = ClassificationHead(C, num_classes=num_classes)

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        high_res_out = self.backbone(images)
        logits = self.classification_head(high_res_out)
        return logits


if __name__ == "__main__":
    net = ClassificationHRNet(17)
    x = torch.randn((16, 3, 224, 224))
    out = net(x)
    print(out.shape)
