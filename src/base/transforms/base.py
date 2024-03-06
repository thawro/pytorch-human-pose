import torchvision.transforms as T
from torch import Tensor


class ImageTransform:
    def __init__(
        self,
        out_size: int | tuple[int, int],
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.out_size = out_size
        self.normalize = T.Normalize(mean=mean, std=std)
        self.unnormalize = UnNormalize(mean=mean, std=std)


class UnNormalize(object):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(x, self.mean, self.std):
            t.mul_(s).add_(m)
        return x
