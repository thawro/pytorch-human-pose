import numpy as np
import torchvision.transforms as T
from torch import Tensor

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

_norm = list[float] | np.ndarray


def list2array(x: _norm) -> np.ndarray:
    if isinstance(x, list):
        return np.array(x)
    return x


class ImageTransform:
    def __init__(
        self,
        out_size: int | tuple[int, int],
        mean: _norm = [0.485, 0.456, 0.406],
        std: _norm = [0.229, 0.224, 0.225],
    ):
        mean = list2array(mean)
        std = list2array(std)
        self.mean = mean
        self.std = std
        self.out_size = out_size
        self.normalize = T.Normalize(mean=mean, std=std)
        self.unnormalize = UnNormalize(mean=mean, std=std)

    @classmethod
    def inverse_transform(
        cls, image: Tensor | np.ndarray, mean: _norm = MEAN, std: _norm = STD
    ) -> np.ndarray:
        if isinstance(image, Tensor):
            image_npy = image.detach().cpu().numpy()
        image_npy = image_npy.transpose(1, 2, 0)
        image_npy = (image_npy * list2array(std)) + list2array(mean)
        image_npy = (image_npy * 255).astype(np.uint8)
        return image_npy


class UnNormalize(object):
    def __init__(self, mean: _norm, std: _norm):
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
