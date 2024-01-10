import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import random

from src.base.transforms import _normalize

keypoint_params = A.KeypointParams(
    format="xy", label_fields=["visibilities"], remove_invisible=False
)

additional_targets = {"masks": "masks"}

compose_params = dict(
    keypoint_params=keypoint_params, additional_targets=additional_targets
)


class SymmetricKeypointsHorizontalFlip:
    def __init__(self, symmetric_keypoints: list[int], p: float = 0.5):
        self.symmetric_keypoints = symmetric_keypoints
        self.p = p

    def __call__(
        self,
        num_obj: int,
        image: np.ndarray,
        masks: list[np.ndarray],
        keypoints: list[tuple[int, int]],
        visibilities: list[list[int]],
    ):
        if random.random() < self.p:
            h, w = image.shape[:2]
            keypoints = [list(kpt) for kpt in keypoints]
            # horizontal flip
            image = np.fliplr(image)
            masks = [np.fliplr(mask).copy() for mask in masks]

            for k in range(len(keypoints)):
                keypoints[k][0] = abs(keypoints[k][0] - w)

            _keypoints = np.array(keypoints).reshape(num_obj, -1, 2)
            _visibilities = np.array(visibilities).reshape(num_obj, -1)

            for j in range(num_obj):
                _keypoints[j] = _keypoints[j][self.symmetric_keypoints]
                _visibilities[j] = _visibilities[j][self.symmetric_keypoints]

            keypoints = _keypoints.reshape(-1, 2).tolist()
            visibilities = _visibilities.reshape(-1, 1).tolist()
        return {
            "image": image,
            "masks": masks,
            "keypoints": keypoints,
            "visibilities": visibilities,
        }


class KeypointsTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        preprocessing: list[A.BasicTransform],
        random: A.Compose,
        inference: A.Compose,
        symmetric_keypoints: list[int] | None,
        out_size: tuple[int, int] = (256, 192),
    ):
        if isinstance(mean, (float, int)):
            mean = [mean] * 3

        if isinstance(std, (float, int)):
            std = [std] * 3

        self.std = np.array(std) * 255
        self.mean = np.array(mean) * 255
        self.out_size = out_size

        self.preprocessing = A.Compose(
            [A.Normalize(mean, std, max_pixel_value=255), *preprocessing],
            **compose_params,
        )

        if symmetric_keypoints is not None:
            self.horizontal_flip = SymmetricKeypointsHorizontalFlip(
                symmetric_keypoints, p=0.5
            )
        else:
            self.horizontal_flip = None

        self.random = random
        self.inference = inference
        self.postprocessing = A.Compose([ToTensorV2()], **compose_params)

    @property
    def inverse_preprocessing(self):
        def transform(image: np.ndarray | Image.Image):
            """Apply inverse of preprocessing to the image (for visualization purposes)."""
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(image, Image.Image):
                image = np.array(image)
            if len(image.shape) == 3:
                _image = image.transpose(1, 2, 0)
            else:
                _image = image.transpose(0, 2, 3, 1)
            _image = _image * self.std + self.mean
            return _image.astype(np.uint8)

        return transform


class SPPEKeypointsTransform(KeypointsTransform):
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        symmetric_keypoints: list[int] | None,
        input_size: tuple[int, int] = (256, 256),
        out_size: tuple[int, int] = (256, 192),
    ):
        input_h, input_w = input_size
        h, w = out_size
        xmin = (input_h - w) // 2
        ymin = (input_w - h) // 2
        xmax = xmin + w
        ymax = ymin + h

        preprocessing = [
            A.Crop(xmin, ymin, xmax, ymax, p=1),
        ]

        random = A.Compose(
            [
                A.Affine(scale=(0.75, 1.25), rotate=(-30, 30), keep_ratio=True, p=0.7),
            ],
            **compose_params,
        )

        inference = A.Compose([], **compose_params)

        super().__init__(
            mean, std, preprocessing, random, inference, symmetric_keypoints, out_size
        )


class MPPEKeypointsTransform(KeypointsTransform):
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        symmetric_keypoints: list[int] | None,
        out_size: tuple[int, int],
    ):
        preprocessing = []

        random = A.Compose(
            [
                A.LongestMaxSize(max(out_size)),
                A.PadIfNeeded(*out_size, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(
                    scale=(0.75, 1.5),
                    rotate=(-30, 30),
                    translate_px=[-40, 40],
                    keep_ratio=True,
                    p=0.7,
                ),
            ],
            **compose_params,
        )

        inference = A.Compose(
            [
                A.LongestMaxSize(max(out_size)),
                A.PadIfNeeded(*out_size, border_mode=cv2.BORDER_CONSTANT),
            ],
            **compose_params,
        )

        super().__init__(
            mean, std, preprocessing, random, inference, symmetric_keypoints, out_size
        )
