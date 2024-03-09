import random
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from src.base.transforms.base import ImageTransform

COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class ComposeKeypointsTransform(object):
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, mask_list: list[np.ndarray], joints_list: list[np.ndarray]
    ) -> tuple[torch.Tensor | np.ndarray, list[np.ndarray], list[np.ndarray]]:
        assert isinstance(mask_list, list)
        assert isinstance(joints_list, list)
        assert len(mask_list) == len(joints_list)
        for t in self.transforms:
            image, mask_list, joints_list = t(image, mask_list, joints_list)
        return image, mask_list, joints_list

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(
        self, image: np.ndarray, mask_list: list[np.ndarray], joints_list: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        return F.to_tensor(image), mask_list, joints_list


class Normalize(object):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, mask_list: list[np.ndarray], joints_list: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask_list, joints_list


class RandomHorizontalFlip(object):
    def __init__(self, flip_index: list[int], hm_sizes: list[int], p: float = 0.5):
        self.flip_index = flip_index
        self.p = p
        self.hm_sizes = hm_sizes

    def __call__(
        self, image: np.ndarray, mask_list: list[np.ndarray], joints_list: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        if random.random() < self.p:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, hm_size in enumerate(self.hm_sizes):
                mask_list[i] = mask_list[i][:, ::-1] - np.zeros_like(mask_list[i])
                joints_list[i] = joints_list[i][:, self.flip_index]
                joints_list[i][:, :, 0] = hm_size - joints_list[i][:, :, 0] - 1

        return image, mask_list, joints_list


class RandomAffineTransform(object):
    def __init__(
        self,
        out_size: int,
        hm_sizes: list[int],
        max_rotation: int = 0,
        min_scale: float = 1,
        max_scale: float = 1,
        scale_type: Literal["short", "long"] = "short",
        max_translate: int = 0,
    ):
        self.out_size = out_size
        self.hm_sizes = hm_sizes
        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        assert scale_type in ["short", "long"], f"Unkonw scale type: {self.scale_type}"

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
        t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(
            shape
        )

    def __call__(
        self, image: np.ndarray, mask_list: list[np.ndarray], joints_list: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        assert isinstance(mask_list, list)
        assert isinstance(joints_list, list)
        assert len(mask_list) == len(joints_list)
        assert len(mask_list) == len(self.hm_sizes)

        height, width = image.shape[:2]

        center = np.array((width / 2, height / 2))
        if self.scale_type == "long":
            scale = max(height, width) / 200
        elif self.scale_type == "short":
            scale = min(height, width) / 200
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            _max_translate = int(self.max_translate * scale)
            dx = np.random.randint(-_max_translate, _max_translate)
            dy = np.random.randint(-_max_translate, _max_translate)
            center[0] += dx
            center[1] += dy

        for i, hm_size in enumerate(self.hm_sizes):
            mat_output = self._get_affine_matrix(center, scale, (hm_size, hm_size), aug_rot)[:2]
            mask_list[i] = (
                cv2.warpAffine(
                    (mask_list[i] * 255).astype(np.uint8), mat_output, (hm_size, hm_size)
                )
                / 255
            )
            mask_list[i] = (mask_list[i] > 0.5).astype(np.float32)

            joints_list[i][:, :, 0:2] = self._affine_joints(joints_list[i][:, :, 0:2], mat_output)

        mat_input = self._get_affine_matrix(center, scale, (self.out_size, self.out_size), aug_rot)[
            :2
        ]
        image = cv2.warpAffine(image, mat_input, (self.out_size, self.out_size))

        return image, mask_list, joints_list


class KeypointsTransform(ImageTransform):
    def __init__(
        self,
        out_size: int,
        hm_resolutions: list[float],
        max_rotation: int = 30,
        min_scale: float = 0.75,
        max_scale: float = 1.5,
        scale_type: Literal["short", "long"] = "short",
        max_translate: int = 40,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__(out_size=out_size, mean=mean, std=std)
        hm_sizes = [int(hm_resolution * out_size) for hm_resolution in hm_resolutions]
        self.train = ComposeKeypointsTransform(
            [
                RandomAffineTransform(
                    out_size=out_size,
                    hm_sizes=hm_sizes,
                    max_rotation=max_rotation,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_type=scale_type,
                    max_translate=max_translate,
                ),
                RandomHorizontalFlip(COCO_FLIP_INDEX, hm_sizes, 0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
        self.inference = ComposeKeypointsTransform(
            [
                RandomAffineTransform(
                    out_size=out_size,
                    hm_sizes=hm_sizes,
                    max_rotation=0,
                    min_scale=1,
                    max_scale=1,
                    scale_type=scale_type,
                    max_translate=0,
                ),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
