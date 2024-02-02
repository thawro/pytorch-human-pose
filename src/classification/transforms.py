import numpy as np
import albumentations as A
import cv2

from src.base.transforms.transforms import _normalize, ImageTransform


class ClassificationTransform(ImageTransform):
    def __init__(
        self, mean: _normalize, std: _normalize, out_size: tuple[int, int] = (224, 224)
    ):
        fill_value = (np.array(mean) * 255).astype(np.uint8).tolist()
        preprocessing = []
        random = A.Compose(
            [
                A.Affine(
                    scale=(0.75, 1.25),
                    rotate=(-30, 30),
                    keep_ratio=True,
                    p=0.7,
                    mode=cv2.BORDER_CONSTANT,
                    cval=fill_value,
                ),
            ],
        )

        inference = A.Compose([])
        self.out_size = out_size

        super().__init__(mean, std, preprocessing, random, inference)
