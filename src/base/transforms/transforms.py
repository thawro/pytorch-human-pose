"""Base transforms"""
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import torch

_normalize = tuple[float, float, float] | list[float] | float | int


class ImageTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        preprocessing: list,
        random: A.Compose,
        inference: A.Compose,
    ):
        if isinstance(mean, (float, int)):
            mean = [mean] * 3

        if isinstance(std, (float, int)):
            std = [std] * 3

        self.mean = np.array(mean) * 255
        self.std = np.array(std) * 255
        self.preprocessing = A.Compose(
            [A.Normalize(mean, std, max_pixel_value=255), *preprocessing],
        )
        self.random = random
        self.inference = inference
        self.postprocessing = A.Compose([ToTensorV2(transpose_mask=True)])

    @property
    def inverse_preprocessing(self):
        def transform(image: np.ndarray | Image.Image | torch.Tensor):
            """Apply inverse of preprocessing to the image (for visualization purposes)."""
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(image, Image.Image):
                image = np.array(image)
            if len(image.shape) == 3:
                _image = image.transpose(1, 2, 0)
            else:
                _image = image.transpose(0, 2, 3, 1)
            _image = _image * self.std + self.mean
            return _image.astype(np.uint8)

        return transform
