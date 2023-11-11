"""Transforms for segmentation task"""

import torchvision.transforms as T

from src.data.transforms.base import ImageDatasetTransform, _normalize


class SegmentationTransform(ImageDatasetTransform):
    def __init__(
        self,
        input_size: int,
        mean: _normalize,
        std: _normalize,
    ):
        self.input_size = input_size
        train = T.Compose([T.RandomHorizontalFlip(p=0.3)])
        inference = T.Compose([])
        super().__init__(mean, std, train, inference)
