import numpy as np
import albumentations as A
import cv2

from src.base.transforms.transforms import _normalize, ImageTransform


class ClassificationTransform(ImageTransform):
    def __init__(
        self,
        out_size: tuple[int, int] = (224, 224),
        mean: _normalize = [0.485, 0.456, 0.406],
        std: _normalize = [0.229, 0.224, 0.225],
    ):
        fill_value = (np.array(mean) * 255).astype(np.uint8).tolist()
        preprocessing = []
        random = A.Compose(
            [
                A.SmallestMaxSize(max(out_size)),
                A.Affine(
                    scale=(0.75, 1.25),
                    rotate=(-30, 30),
                    keep_ratio=True,
                    p=0.7,
                    mode=cv2.BORDER_CONSTANT,
                    cval=fill_value,
                ),
                A.RandomCrop(*out_size),
            ],
        )

        inference = A.Compose(
            [
                A.LongestMaxSize(max(out_size)),
                A.PadIfNeeded(
                    *out_size, border_mode=cv2.BORDER_CONSTANT, value=fill_value
                ),
            ]
        )
        self.out_size = out_size

        super().__init__(preprocessing, random, inference, mean, std)
