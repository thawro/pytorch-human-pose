import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.base.transforms import _normalize


class KeypointsTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        size: int = 512,
        multi_obj: bool = False,
    ):
        self.std = np.array(std) * 255
        self.mean = np.array(mean) * 255
        self.size = size
        self.multi_obj = multi_obj

        keypoint_params = A.KeypointParams(
            format="xy", label_fields=["visibilities"], remove_invisible=False
        )

        self.preprocessing = A.Compose([A.Normalize(mean, std, max_pixel_value=255)])
        self.random = A.Compose(
            [A.Rotate(limit=20, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)],
            keypoint_params=keypoint_params,
        )

        if multi_obj:
            smallest_max_size = A.SmallestMaxSize(size)
            train_postprocessing = A.Compose(
                [smallest_max_size, A.RandomCrop(size, size)]
            )
            inference_postprocessing = A.Compose(
                [smallest_max_size, A.CenterCrop(size, size)]
            )
            postprocessing = []

        else:
            postprocessing = [
                A.LongestMaxSize(size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            ]

        postprocessing.append(ToTensorV2(transpose_mask=True))
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
