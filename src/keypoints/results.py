from dataclasses import dataclass, field
import numpy as np
from torch import Tensor
from src.keypoints.grouping import SPPEHeatmapParser, MPPEHeatmapParser
from src.keypoints.metrics import OKS, PCKh, object_PCKh, object_OKS, EvaluationMetric
from typing import Callable
from functools import partial
import torchvision.transforms.functional as F
import torch


def match_preds_to_targets(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    extra_coords: list,
    match_fn: Callable,
) -> np.ndarray:
    # pred_kpts shape: [num_obj_pred, num_kpts, 3]
    # 3 for: x, y, score
    # target_kpts shape: [num_obj_target, num_kpts, 2]
    # 2 for: x, y
    # extra_coords is list of ground truth extra_coords (for size calculation)
    # COCO: area segmentation, MPII: head_xyxy coords
    num_target_obj, num_kpts = target_kpts.shape[:2]
    pred_obj_scores = pred_kpts[..., 2].mean(-1)
    sorted_idxs = np.argsort(pred_obj_scores, kind="mergesort")
    target_matches_idx = [-1 for _ in range(num_target_obj)]
    target_matches_vals = [0 for _ in range(num_target_obj)]
    matched_idxs = []
    for pred_idx in sorted_idxs:
        p_kpts = pred_kpts[pred_idx]
        for target_idx, t_kpts in enumerate(target_kpts):
            if target_idx in matched_idxs:
                continue
            match_val = match_fn(
                p_kpts[..., :2], t_kpts[..., :2], extra_coords[target_idx]
            )
            if match_val > target_matches_vals[target_idx]:
                target_matches_vals[target_idx] = match_val
                target_matches_idx[target_idx] = pred_idx
                matched_idxs.append(target_idx)
    # remove duplicates
    # target_matches_idx = list(set(target_matches_idx))
    return pred_kpts[target_matches_idx]


@dataclass
class KeypointsResults:
    images: np.ndarray
    target_heatmaps: np.ndarray
    pred_heatmaps: np.ndarray
    target_keypoints: np.ndarray
    pred_keypoints: list[np.ndarray]
    pred_scores: list[np.ndarray]
    extra_coords: list
    match_preds_to_targets: Callable = field(init=False)
    metric: EvaluationMetric = field(init=False)

    def evaluate(self) -> dict[str, float]:
        return self.metric.evaluate_results(
            self.pred_keypoints, self.target_keypoints, self.extra_coords
        )


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
        h, w = images.shape[1:3]
        pred_heatmaps = F.resize(pred_heatmaps, [h, w], antialias=True)
        batch_size, num_kpts = pred_heatmaps.shape[:2]
        parser = SPPEHeatmapParser(num_kpts, det_thr=det_thr)

        pred_joints = []
        for i in range(batch_size):
            _heatmaps = pred_heatmaps[i]
            parsed_joints = parser.parse(_heatmaps.unsqueeze(0))
            joints = cls.match_preds_to_targets(
                parsed_joints, target_keypoints[i], extra_coords[i]
            )
            pred_joints.append(joints)
        # pred_joints = np.stack(pred_joints)
        pred_kpts_coords = [joints[..., :2] for joints in pred_joints]
        pred_kpts_scores = [joints[..., 2] for joints in pred_joints]

        npy_pred_heatmaps = pred_heatmaps.detach().cpu().numpy()

        return cls(
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
        stages_target_heatmaps: list[Tensor],
        stages_pred_heatmaps: list[Tensor],
        target_keypoints: np.ndarray,
        extra_coords: list,
        det_thr: float = 0.2,
        tag_thr: float = 1,
    ) -> "MPPEKeypointsResults":
        h, w = images.shape[1:3]
        num_stages = len(stages_target_heatmaps)
        num_kpts = stages_target_heatmaps[0].shape[1]

        # Average across resolutions
        pred_heatmaps = []
        pred_tags = []
        for i in range(num_stages):
            stage_pred_heatmaps = F.resize(
                stages_pred_heatmaps[i], [h, w], antialias=True
            )
            kpts_heatmaps = stage_pred_heatmaps[:, :num_kpts]
            tags_heatmaps = stage_pred_heatmaps[:, num_kpts:]
            pred_heatmaps.append(kpts_heatmaps)
            pred_tags.append(tags_heatmaps)

        pred_heatmaps = torch.stack(pred_heatmaps).mean(0)
        pred_tags = torch.stack(pred_tags).mean(0)

        batch_size, num_kpts = pred_heatmaps.shape[:2]
        parser = MPPEHeatmapParser(num_kpts, det_thr=det_thr, tag_thr=tag_thr)

        pred_joints = []
        for i in range(batch_size):
            _heatmaps = pred_heatmaps[i]
            _tags = pred_tags[i]
            parsed_joints = parser.parse(
                _heatmaps.unsqueeze(0), _tags.unsqueeze(0), adjust=True, refine=True
            )
            joints = cls.match_preds_to_targets(
                parsed_joints, target_keypoints[i], extra_coords[i]
            )
            pred_joints.append(joints)
        # pred_joints = np.stack(pred_joints)
        pred_kpts_coords = [joints[..., :2] for joints in pred_joints]
        pred_kpts_scores = [joints[..., 2] for joints in pred_joints]

        npy_pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
        npy_pred_tags = pred_tags.detach().cpu().numpy()

        return cls(
            images,
            stages_target_heatmaps[-1].cpu().numpy(),
            npy_pred_heatmaps,
            target_keypoints,
            pred_kpts_coords,
            pred_kpts_scores,
            extra_coords,
            npy_pred_tags,
        )


match_mpii_preds_to_targets = partial(match_preds_to_targets, match_fn=object_PCKh)
match_coco_preds_to_targets = partial(match_preds_to_targets, match_fn=object_OKS)


class SppeMpiiKeypointsResults(SPPEKeypointsResults):
    match_preds_to_targets = match_mpii_preds_to_targets
    metric = PCKh(alpha=0.5)


class SppeCocoKeypointsResults(SPPEKeypointsResults):
    match_preds_to_targets = match_coco_preds_to_targets
    metric = OKS()


class MppeMpiiKeypointsResults(MPPEKeypointsResults):
    match_preds_to_targets = match_mpii_preds_to_targets
    metric = PCKh(alpha=0.5)


class MppeCocoKeypointsResults(MPPEKeypointsResults):
    match_preds_to_targets = match_coco_preds_to_targets
    metric = OKS()
