from dataclasses import dataclass
import numpy as np
from torch import Tensor
from .ops import get_sppe_kpts_coords, get_mppe_ae_kpts_coords
import torchvision.transforms.functional as F


@dataclass
class SPPEKeypointsResults:
    images: np.ndarray
    pred_heatmaps: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_preds(cls, images: np.ndarray, heatmaps: Tensor) -> "SPPEKeypointsResults":
        h, w = images.shape[1:3]
        heatmaps = F.resize(heatmaps, [h, w])
        numpy_heatmaps = heatmaps.detach().cpu().numpy()

        kpts_coords, kpts_scores = get_sppe_kpts_coords(
            numpy_heatmaps, return_scores=True
        )

        kpts_coords = kpts_coords.numpy()
        kpts_scores = kpts_scores.numpy()
        return SPPEKeypointsResults(images, numpy_heatmaps, kpts_coords, kpts_scores)


@dataclass
class MPPEKeypointsResults:
    images: np.ndarray
    pred_heatmaps: np.ndarray
    pred_tags: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_preds(
        cls, images: np.ndarray, heatmaps: Tensor, tags: Tensor
    ) -> "MPPEKeypointsResults":
        h, w = images.shape[1:3]
        heatmaps = F.resize(heatmaps, [h, w])
        numpy_heatmaps = heatmaps.detach().cpu().numpy()
        numpy_tags = tags.detach().cpu().numpy()

        kpts_coords, kpts_scores = get_mppe_ae_kpts_coords(
            numpy_heatmaps, numpy_tags, return_scores=True
        )

        kpts_coords = kpts_coords.numpy()
        kpts_scores = kpts_scores.numpy()
        return MPPEKeypointsResults(
            images, numpy_heatmaps, numpy_tags, kpts_coords, kpts_scores
        )
