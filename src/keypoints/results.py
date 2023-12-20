from dataclasses import dataclass
import numpy as np
from torch import Tensor
import torchvision.transforms.functional as F
from src.keypoints.grouping import SPPEHeatmapParser, MPPEHeatmapParser


@dataclass
class KeypointsResults:
    images: np.ndarray
    target_heatmaps: np.ndarray
    pred_heatmaps: np.ndarray
    target_keypoints: np.ndarray
    pred_keypoints: np.ndarray
    pred_scores: np.ndarray
    extra_coords: list


@dataclass
class SPPEKeypointsResults(KeypointsResults):
    @classmethod
    def from_preds(
        cls,
        images: np.ndarray,
        target_heatmaps: Tensor,
        pred_heatmaps: Tensor,
        target_keypoints: list,
        extra_coords: list,
        det_thr: float = 0.2,
    ) -> "SPPEKeypointsResults":
        num_kpts = pred_heatmaps.shape[1]
        parser = SPPEHeatmapParser(num_kpts, det_thr=det_thr)
        h, w = images.shape[1:3]
        pred_heatmaps = F.resize(pred_heatmaps, [h, w])
        all_joints = []
        for heatmap in pred_heatmaps:
            _heatmaps = heatmap.unsqueeze(0)
            joints = parser.parse(_heatmaps)
            all_joints.append(joints)
        all_joints = np.stack(all_joints)
        pred_kpts_coords = all_joints[..., :2]
        pred_kpts_scores = all_joints[..., 2]

        npy_pred_heatmaps = pred_heatmaps.detach().cpu().numpy()

        return SPPEKeypointsResults(
            images,
            target_heatmaps.cpu().numpy(),
            npy_pred_heatmaps,
            np.array(target_keypoints),
            pred_kpts_coords,
            pred_kpts_scores,
            extra_coords,
        )


@dataclass
class MPPEKeypointsResults(KeypointsResults):
    pred_tags: np.ndarray

    @classmethod
    def from_preds(
        cls,
        images: np.ndarray,
        target_heatmaps: Tensor,
        pred_heatmaps: Tensor,
        target_keypoints: np.ndarray,
        extra_coords: list,
        pred_tags: Tensor,
        det_thr: float = 0.2,
        tag_thr: float = 1,
    ) -> "MPPEKeypointsResults":
        num_kpts = pred_heatmaps.shape[1]
        parser = MPPEHeatmapParser(num_kpts, det_thr=det_thr, tag_thr=tag_thr)

        h, w = images.shape[1:3]
        pred_heatmaps = F.resize(pred_heatmaps, [h, w])

        all_joints = []
        for heatmap, tag in zip(pred_heatmaps, pred_tags):
            _heatmaps = heatmap.unsqueeze(0)
            _tags = tag.unsqueeze(0)
            joints = parser.parse(_heatmaps, _tags, adjust=True, refine=True)
            all_joints.append(joints)
        all_joints = np.stack(all_joints)
        pred_kpts_coords = all_joints[..., :2]
        pred_kpts_scores = all_joints[..., 2]

        npy_pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
        npy_pred_tags = pred_tags.detach().cpu().numpy()

        return MPPEKeypointsResults(
            images,
            target_heatmaps.numpy(),
            npy_pred_heatmaps,
            target_keypoints,
            pred_kpts_coords,
            pred_kpts_scores,
            extra_coords,
            npy_pred_tags,
        )
