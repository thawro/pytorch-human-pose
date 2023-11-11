"""Transforms for segmentation task"""
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
from PIL import Image
from abc import abstractmethod

_normalize = tuple[float, float, float]


class InversableTransform:
    @abstractmethod
    def __call__(self, image: Tensor):
        pass

    @abstractmethod
    def inverse(self, image: Tensor) -> Tensor:
        pass


class InversableCompose:
    def __init__(self, transforms: list[InversableTransform]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray | Tensor) -> tuple[Tensor, dict]:
        kwargs = {}
        for transform in self.transforms:
            image, transform_kwargs = transform(image, return_kwargs=True)
            kwargs.update(transform_kwargs)
        return image, kwargs

    def inverse(self, image: Tensor, **kwargs) -> np.ndarray | Tensor:
        inverse_image = image
        for transform in reversed(self.transforms):
            inverse_image, *_ = transform.inverse(inverse_image, **kwargs)
        return inverse_image


class ImageTransform:
    def __init__(self, mean: _normalize, std: _normalize, transform: T.Compose):
        self.mean = mean
        self.std = std
        self.preprocessing = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.transform = T.Compose([self.preprocessing, transform])

    def __call__(self, image: np.ndarray):
        return self.transform(image)

    @property
    def inverse_preprocessing(self):
        def transform(image: np.ndarray | Image.Image):
            """Apply inverse of preprocessing to the image (for visualization purposes)."""
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(image, Image.Image):
                image = np.array(image)
            _image = image.transpose(1, 2, 0)
            _image = (_image * np.array(self.std)) + np.array(self.mean)
            return _image

        return transform


class DatasetTransform:
    def __init__(self, train_transform: T.Compose, infernece_transform: T.Compose):
        self.train = train_transform
        self.inference = infernece_transform


class ImageDatasetTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        train_transform: T.Compose,
        infernece_transform: T.Compose,
    ):
        self.train = ImageTransform(mean, std, train_transform)
        self.inference = ImageTransform(mean, std, infernece_transform)
