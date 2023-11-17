"""Base transforms"""
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import torch

_normalize = tuple[float, float, float] | list[float]


class ImageTransform:
    def __init__(
        self, mean: _normalize, std: _normalize, transform: A.Compose, **kwargs
    ):
        self.mean = np.array(mean) * 255
        self.std = np.array(std) * 255
        self.transform = A.Compose(
            [
                A.Normalize(mean, std, max_pixel_value=255),
                transform,
                ToTensorV2(transpose_mask=True),
            ],
            **kwargs
        )

    def __call__(self, image: Image.Image, **kwargs):
        return self.transform(image=image, **kwargs)


class BaseTransform:
    def __init__(self):
        self.preprocessing = A.Compose([])
        self.random = A.Compose([])
        self.postprocessing = A.Compose([ToTensorV2(transpose_mask=True)])

    @property
    def inverse_preprocessing(self):
        def transform(x: np.ndarray | Image.Image):
            """Apply inverse of preprocessing to the image (for visualization purposes)."""
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(x, Image.Image):
                x = np.array(x)
            return x

        return transform
