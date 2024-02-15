import numpy as np
import albumentations as A
import cv2

from src.base.transforms.transforms import _normalize, ImageTransform
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T


class ClassificationTransform(ImageTransform):
    def __init__(
        self,
        out_size: tuple[int, int] = (224, 224),
        mean: _normalize = [0.485, 0.456, 0.406],
        std: _normalize = [0.229, 0.224, 0.225],
    ):
        preprocessing = []

        random = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(out_size[0], antialias=True),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        inference = T.Compose(
            [
                T.ToTensor(),
                T.Resize(int(out_size[0] / 0.875), antialias=True),
                T.CenterCrop(out_size[0]),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.out_size = out_size

        super().__init__(preprocessing, random, inference, mean, std)
