from typing import Literal
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from src.data.transforms.base import InversableTransform

_normalize = tuple[float, float, float]
_padding = tuple[int, int, int, int]
_size = tuple[int, int]
_crop_coords = tuple[int, int, int, int]


class Pad(InversableTransform):
    def __call__(
        self,
        image: Tensor,
        pad_x: int,
        pad_y: int,
        return_kwargs: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, _padding]]:
        pad_left = pad_x // 2
        pad_top = pad_y // 2

        pad_right = pad_x - pad_left
        pad_bottom = pad_y - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        padded_image = F.pad(image, padding)
        if return_kwargs:
            return padded_image, {"padding": padding}
        return padded_image

    def inverse(self, image: Tensor, padding: _padding, **kwargs) -> Tensor:
        pad_left, pad_top, pad_right, pad_bottom = padding
        return image[..., pad_top:-pad_bottom, pad_left:-pad_right]


class SquarePad(Pad):
    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _padding]]:
        h, w = image.shape[-2:]
        max_wh = np.max([w, h])
        pad_x = max_wh - w
        pad_y = max_wh - h
        return super().__call__(image, pad_x, pad_y, return_kwargs)


class PadIfNeeded(Pad):
    def __init__(self, size: _size):
        self.h, self.w = size

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _padding]]:
        h, w = image.shape[-2:]
        pad_y = max(self.h - h, 0)
        pad_x = max(self.w - w, 0)
        return super().__call__(image, pad_x, pad_y, return_kwargs)


class CenterCrop(InversableTransform):
    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _crop_coords | _size]]:
        h, w = image.shape[-2:]
        crop_size = int(self.size)
        crop_y = (h - crop_size) // 2
        crop_x = (w - crop_size) // 2
        xmin = crop_x
        xmax = xmin + self.size
        ymin = crop_y
        ymax = ymin + self.size
        cropped_image = image[..., ymin:ymax, xmin:xmax]
        crop_coords = (xmin, ymin, xmax, ymax)
        crop_orig_size = (h, w)
        if return_kwargs:
            return cropped_image, {
                "crop_coords": crop_coords,
                "crop_orig_size": crop_orig_size,
            }
        return cropped_image

    def inverse(
        self, image: Tensor, crop_coords: _crop_coords, crop_orig_size: _size, **kwargs
    ) -> Tensor:
        h, w = crop_orig_size
        xmin, ymin, xmax, ymax = crop_coords
        blank = torch.zeros(3, h, w, dtype=image.dtype, device=image.device)
        blank[..., ymin:ymax, xmin:xmax] = image
        return blank


class SmallestMaxSize(InversableTransform):
    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _size]]:
        h, w = image.shape[-2:]
        aspect_ratio = h / w

        if h > w:
            new_w = int(self.size)
            new_h = int(new_w * aspect_ratio)
        else:
            new_h = int(self.size)
            new_w = int(new_h / aspect_ratio)
        new_size = (new_h, new_w)
        resized_image = F.resize(
            image, new_size, antialias=False
        )  # TODO: onnx doesnt allow antialias=True
        resize_orig_size = (h, w)
        if return_kwargs:
            return resized_image, {"resize_orig_size": resize_orig_size}
        return resized_image

    def inverse(self, image: Tensor, resize_orig_size: _size, **kwargs) -> Tensor:
        return F.resize(image, resize_orig_size, antialias=False)


class HeightMaxSize(InversableTransform):
    def __init__(self, size: int):
        self.size = torch.tensor(size)

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _size]]:
        h, w = image.shape[-2:]
        aspect_ratio = h / w

        new_h = self.size
        new_w = new_h / aspect_ratio
        new_size = torch.stack([new_h, new_w]).to(torch.int32)
        resized_image = F.resize(
            image, new_size, antialias=False
        )  # TODO: onnx doesnt allow antialias=True
        resize_orig_size = (h, w)
        if return_kwargs:
            return resized_image, {"resize_orig_size": resize_orig_size}
        return resized_image

    def inverse(self, image: Tensor, resize_orig_size: _size, **kwargs) -> Tensor:
        return F.resize(image, resize_orig_size, antialias=False)


class LongestMaxSize(InversableTransform):
    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, _size]]:
        h, w = image.shape[-2:]
        aspect_ratio = h / w

        if h > w:
            new_h = int(self.size)
            new_w = int(new_h / aspect_ratio)
        else:
            new_w = int(self.size)
            new_h = int(new_w * aspect_ratio)
        new_size = (new_h, new_w)
        resized_image = F.resize(image, new_size, antialias=True)
        resize_orig_size = (h, w)
        if return_kwargs:
            return resized_image, {"resize_orig_size": resize_orig_size}
        return resized_image

    def inverse(self, image: Tensor, resize_orig_size: _size, **kwargs) -> Tensor:
        return F.resize(image, resize_orig_size, antialias=True)


class Normalize(InversableTransform):
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        max_pixel_value: Literal[1, 255],
    ):
        self.mean = torch.tensor(mean) * max_pixel_value
        self.std = torch.tensor(std) * max_pixel_value

        if max_pixel_value == 255:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.float32

    def __call__(
        self, image: Tensor, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict]:
        normalized_image = (image - self.mean) / self.std
        if return_kwargs:
            return normalized_image, {}
        return normalized_image

    def inverse(self, image: Tensor, **kwargs) -> Tensor:
        return ((image * self.std) + self.mean).to(self.dtype)


class ToTensor(InversableTransform):
    def __call__(
        self, image: np.ndarray, return_kwargs: bool = False
    ) -> Tensor | tuple[Tensor, dict]:
        tensor_image = torch.from_numpy(image).permute(2, 0, 1)
        if return_kwargs:
            return tensor_image, {}
        return tensor_image

    def inverse(self, image: Tensor, **kwargs) -> np.ndarray:
        return image.permute(1, 2, 0).numpy()
