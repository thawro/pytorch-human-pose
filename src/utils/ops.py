import numpy as np
import cv2
import torch
from torch import Tensor
import math
import torch.nn.functional as F


def keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    img = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)

    # Output
    out_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out_mask, [largest_contour], -1, 1, cv2.FILLED)
    out_mask = cv2.bitwise_and(mask.astype(np.uint8), out_mask).astype(np.float32)
    return out_mask


def gaussian_kernel_2d(
    kernel_size: int, sigma: float, n_channels: int = 1, device: str = "cuda"
) -> Tensor:
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.0
    var = sigma**2.0

    kernel = (1 / (2 * math.pi * var)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * var)
    )
    return (kernel / torch.sum(kernel)).to(device).repeat(n_channels, n_channels, 1, 1)


def convolve_gaussian_2d(
    image: np.ndarray, kernel_size: int, sigma: float = 3, device: str = "cuda"
) -> Tensor:
    if len(image.shape) == 2:
        n_channels = 1
    else:
        n_channels = image.shape[-1]
    kernel = gaussian_kernel_2d(
        kernel_size, sigma, n_channels=n_channels, device=device
    )
    padding = (kernel_size - 1) // 2
    input_image = torch.from_numpy(image).to(device).unsqueeze(0).unsqueeze(0)
    return F.conv2d(input_image, kernel, groups=1, padding=padding).squeeze()


def avg_pool(frame: np.ndarray, kernel: tuple[int, int] = (4, 4)) -> Tensor:
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame = torch.nn.functional.avg_pool2d(frame, kernel)
    frame = frame.squeeze().permute(1, 2, 0).numpy()
    return frame


def minmax(x: Tensor, dim: int, keepdim: bool, scaler: float) -> Tensor:
    _min = torch.min(x, dim=dim, keepdim=keepdim).values
    _max = torch.max(x, dim=dim, keepdim=keepdim).values
    return (x - _min) / (_max - _min) * scaler


def filter_labels_from_mask(
    mask: Tensor, labels_to_filer: str | list[str], all_labels: list[str]
) -> Tensor:
    new_mask = torch.zeros_like(mask, dtype=torch.float32)
    if isinstance(labels_to_filer, str):
        labels_to_filer = [labels_to_filer]
    for label in labels_to_filer:
        idx = all_labels.index(label) + 1  # +1 for background
        new_mask[mask == idx] = 1
    return new_mask


def find_center_of_mass(binary_mask: np.ndarray) -> tuple[int, int]:
    mask = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    moments = cv2.moments(mask)
    try:
        x_center = int(moments["m10"] / moments["m00"])
        y_center = int(moments["m01"] / moments["m00"])
    except ZeroDivisionError:
        x_center = mask.shape[1] // 2
        y_center = mask.shape[0] // 2
    return int(y_center), int(x_center)
