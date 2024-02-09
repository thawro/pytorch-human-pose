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
            [A.RandomResizedCrop(*out_size), A.HorizontalFlip(p=0.5)],
        )

        inference = A.Compose(
            [A.SmallestMaxSize(max(out_size)), A.CenterCrop(*out_size)]
        )
        self.out_size = out_size

        super().__init__(preprocessing, random, inference, mean, std)
