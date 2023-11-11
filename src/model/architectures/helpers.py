"""Helpers for architectures implementation."""

from torch import Tensor, nn


class ConvBnAct(nn.Module):
    """Conv2d -> BatchNormalization -> Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = True,
        activation: str = "ReLU",
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not batch_norm,
        )
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = getattr(nn, activation)()
        self.batchnorm = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.activation(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Networks https://arxiv.org/abs/1709.01507
    """

    def __init__(self, input_channels=64, reduction_ratio=2):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.excitation = nn.Sequential(
            # excitation
            nn.Linear(in_features=input_channels, out_features=input_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=input_channels // reduction_ratio, out_features=input_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, e):
        squeezed = self.squeeze(e)
        squeezed = squeezed.squeeze(3).squeeze(2)

        excited = self.excitation(squeezed)
        excited = excited.unsqueeze(2).unsqueeze(3)

        return excited * x
