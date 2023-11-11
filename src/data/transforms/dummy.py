"""Transforms for segmentation task"""

import torchvision.transforms as T
from src.data.transforms.base import DatasetTransform


class DummyTransform(DatasetTransform):
    def __init__(self):
        train = T.Compose([])
        inference = T.Compose([])
        super().__init__(train, inference)
