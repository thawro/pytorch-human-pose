import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.base.transforms import _normalize

keypoint_params = A.KeypointParams(
    format="xy", label_fields=["visibilities"], remove_invisible=False
)


class KeypointsTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        random: A.Compose,
        postprocessing: list[A.BasicTransform],
        out_size: tuple[int, int] = (256, 192),
    ):
        self.std = np.array(std) * 255
        self.mean = np.array(mean) * 255
        self.out_size = out_size
        input_h, input_w = 256, 256
        h, w = out_size
        xmin = (input_h - w) // 2
        ymin = (input_w - h) // 2
        xmax = xmin + w
        ymax = ymin + h

        self.preprocessing = A.Compose(
            [
                A.Normalize(mean, std, max_pixel_value=255),
                A.Crop(xmin, ymin, xmax, ymax, p=1),
            ],
            keypoint_params=keypoint_params,
        )

        self.random = random
        self.postprocessing = A.Compose(postprocessing, keypoint_params=keypoint_params)

    @property
    def inverse_preprocessing(self):
        def transform(image: np.ndarray | Image.Image):
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


class SPPEKeypointsTransform(KeypointsTransform):
    def __init__(self, mean: _normalize, std: _normalize, out_size: tuple[int, int]):
        random = A.Compose(
            [
                A.Affine(scale=(0.75, 1.25), rotate=(-30, 30), keep_ratio=True, p=1),
            ],
            keypoint_params=keypoint_params,
        )
        postprocessing = [ToTensorV2()]
        super().__init__(mean, std, random, postprocessing, out_size=out_size)
