from dataclasses import dataclass, field
import numpy as np
from torch import Tensor
from src.keypoints.grouping import SPPEHeatmapParser, MPPEHeatmapParser
from src.keypoints.metrics import OKS, PCKh, object_PCKh, object_OKS, EvaluationMetric
from typing import Callable
from functools import partial
import torchvision.transforms.functional as F
import torch
from src.keypoints.visualization import plot_heatmaps, plot_connections
import cv2
from src.utils.image import make_grid


def match_preds_to_targets(
    pred_joints: np.ndarray,
    target_kpts: np.ndarray,
    target_visibilities: np.ndarray,
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
    pred_obj_scores = pred_joints[..., 2].mean(-1)
    sorted_idxs = np.argsort(pred_obj_scores, kind="mergesort")
    target_matches_idx = [-1 for _ in range(num_target_obj)]
    target_matches_vals = [0 for _ in range(num_target_obj)]
    matched_idxs = []
    for pred_idx in sorted_idxs:
        p_kpts = pred_joints[pred_idx]
        for target_idx in range(len(target_kpts)):
            t_kpts = target_kpts[target_idx]
            t_vis = target_visibilities[target_idx]
            match_val = match_fn(
                p_kpts[..., :2], t_kpts[..., :2], t_vis, extra_coords[target_idx]
            )
            # if target_idx in matched_idxs:
            #     continue
            if match_val > target_matches_vals[target_idx]:
                target_matches_vals[target_idx] = match_val
                target_matches_idx[target_idx] = pred_idx
                matched_idxs.append(target_idx)
    # remove duplicates
    # target_matches_idx = list(set(target_matches_idx))
    return pred_joints[target_matches_idx]


@dataclass
class KeypointsResults:
    images: np.ndarray
    target_heatmaps: np.ndarray
    pred_heatmaps: np.ndarray
    target_keypoints: np.ndarray
    target_visibilities: np.ndarray
    pred_keypoints: list[np.ndarray]
    pred_scores: list[np.ndarray]
    extra_coords: list
    match_preds_to_targets: Callable = field(init=False)
    metric: EvaluationMetric = field(init=False)

    def evaluate(self) -> dict[str, float] | float:
        return self.metric.evaluate_results(
            self.pred_keypoints,
            self.target_keypoints,
            self.target_visibilities,
            self.extra_coords,
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
        target_visibilities: np.ndarray,
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
                parsed_joints,
                target_keypoints[i],
                target_visibilities[i],
                extra_coords[i],
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
            target_visibilities,
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
        stages_pred_kpts_heatmaps: list[Tensor],
        stages_pred_tags_heatmaps: list[Tensor],
        target_keypoints: np.ndarray,
        target_visibilities: np.ndarray,
        extra_coords: list,
        max_num_people: int = 10,
        det_thr: float = 0.1,
        tag_thr: float = 1,
    ) -> "MPPEKeypointsResults":
        h, w = images.shape[1:3]

        stages_pred_kpts_heatmaps = [
            F.resize(hm, [h, w], antialias=True) for hm in stages_pred_kpts_heatmaps
        ]
        stages_pred_tags_heatmaps = [
            F.resize(hm, [h, w], antialias=True) for hm in stages_pred_tags_heatmaps
        ]
        num_kpts = stages_target_heatmaps[0].shape[1]

        pred_kpts_heatmaps = torch.stack(stages_pred_kpts_heatmaps, dim=-1)

        pred_tags_heatmaps = torch.stack(stages_pred_tags_heatmaps, dim=-1)

        batch_size, num_kpts = pred_kpts_heatmaps.shape[:2]
        parser = MPPEHeatmapParser(
            num_kpts, max_num_people=max_num_people, det_thr=det_thr, tag_thr=tag_thr
        )

        kpts_hms_to_parse = torch.nn.functional.sigmoid(pred_kpts_heatmaps.mean(-1))

        pred_joints = []
        for i in range(batch_size):
            _heatmaps = kpts_hms_to_parse[i]

            _tags = pred_tags_heatmaps[i]
            parsed_joints = parser.parse(
                _heatmaps.unsqueeze(0),
                _tags.unsqueeze(0),
                adjust=True,
                refine=True,
            )
            joints = cls.match_preds_to_targets(
                parsed_joints,
                target_keypoints[i],
                target_visibilities[i],
                extra_coords[i],
            )
            pred_joints.append(joints)

        pred_kpts_coords = [joints[..., :2] for joints in pred_joints]
        pred_kpts_scores = [joints[..., 2] for joints in pred_joints]

        npy_pred_heatmaps = pred_kpts_heatmaps.detach().cpu().numpy()
        npy_pred_tags = pred_tags_heatmaps.detach().cpu().numpy()

        return cls(
            images,
            stages_target_heatmaps[-1].cpu().numpy(),
            npy_pred_heatmaps,
            target_keypoints,
            target_visibilities,
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


### Inference


@dataclass
class InferenceMPPEKeypointsResult:
    annot: np.ndarray | None
    input_image: np.ndarray
    image: np.ndarray
    pred_kpts_heatmaps: np.ndarray
    pred_tags_heatmaps: np.ndarray
    pred_keypoints: np.ndarray
    pred_scores: np.ndarray
    limbs: list[tuple[int, int]]

    @classmethod
    def from_preds(
        cls,
        annot,
        input_image: np.ndarray,
        image: np.ndarray,
        scale: float,
        pad: tuple[int, int, int, int],
        stages_pred_kpts_heatmaps: list[Tensor],
        stages_pred_tags_heatmaps: list[Tensor],
        limbs: list[tuple[int, int]],
        max_num_people: int = 10,
        det_thr: float = 0.1,
        tag_thr: float = 1,
    ) -> "InferenceMPPEKeypointsResult":
        h, w = input_image.shape[:2]

        stages_pred_kpts_heatmaps = [
            F.resize(hm, [h, w], antialias=True) for hm in stages_pred_kpts_heatmaps
        ]
        stages_pred_tags_heatmaps = [
            F.resize(hm, [h, w], antialias=True) for hm in stages_pred_tags_heatmaps
        ]
        num_kpts = stages_pred_kpts_heatmaps[0].shape[1]

        pred_kpts_heatmaps = torch.stack(stages_pred_kpts_heatmaps, dim=-1)
        pred_tags_heatmaps = torch.stack(stages_pred_tags_heatmaps, dim=-1)

        if pred_tags_heatmaps.shape[0] > 1:
            pred_tags_heatmaps = [hm for hm in pred_tags_heatmaps]
            pred_tags_heatmaps = torch.concat(pred_tags_heatmaps, dim=-1).unsqueeze(0)

        batch_size, num_kpts = pred_kpts_heatmaps.shape[:2]
        parser = MPPEHeatmapParser(
            num_kpts, max_num_people=max_num_people, det_thr=det_thr, tag_thr=tag_thr
        )

        # kpts_hms_to_parse = torch.nn.functional.sigmoid(pred_kpts_heatmaps).mean(-1)
        kpts_hms_to_parse = torch.nn.functional.sigmoid(pred_kpts_heatmaps.mean(-1))

        joints = parser.parse(
            kpts_hms_to_parse, pred_tags_heatmaps, adjust=True, refine=True
        )

        pred_kpts_coords = joints[..., :2]
        pred_kpts_scores = joints[..., 2]
        scaled_pred_kpts_coords = pred_kpts_coords.copy()
        scaled_pred_kpts_coords[..., 0] -= pad[2]
        scaled_pred_kpts_coords[..., 1] -= pad[0]
        scaled_pred_kpts_coords = (scaled_pred_kpts_coords * scale).astype(np.int32)

        npy_pred_kpts_heatmaps = (
            torch.nn.functional.sigmoid(pred_kpts_heatmaps).cpu().numpy()[0]
        )
        npy_pred_tags_heatmaps = pred_tags_heatmaps.cpu().numpy()[0]

        return cls(
            annot,
            input_image,
            image,
            npy_pred_kpts_heatmaps,
            npy_pred_tags_heatmaps,
            scaled_pred_kpts_coords,
            pred_kpts_scores,
            limbs,
        )

    def plot(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO
        if self.annot is not None:
            objects = self.annot["objects"]
            target_kpts = []  # np.zeros((num_obj, num_kpts, 2), dtype=np.int32)
            target_visibilities = []  # np.zeros((num_obj, num_kpts), dtype=np.int32)
            seg_polygons = []  # [obj["segmentation"] for obj in objects]
            for i in range(len(objects)):
                kpts = objects[i]["keypoints"]

                v = np.array([k["visibility"] for k in kpts])
                if v.sum() > 0:
                    xy = np.array([[k["x"], k["y"]] for k in kpts])
                    target_visibilities.append(v)
                    target_kpts.append(xy)
                    seg_polygons.append(objects[i]["segmentation"])

            if len(target_visibilities) > 0:
                target_visibilities = np.stack(target_visibilities)
                target_kpts = np.stack(target_kpts)
                pred_joints = np.concatenate(
                    [self.pred_keypoints, np.expand_dims(self.pred_scores, -1)],
                    axis=-1,
                )
                pred_joints = match_preds_to_targets(
                    pred_joints,
                    target_kpts,
                    target_visibilities,
                    seg_polygons,
                    object_OKS,
                )
                self.pred_keypoints = pred_joints[..., :2]
                self.pred_scores = pred_joints[..., 2]

                oks = OKS()
                oks_value = oks.image_eval(
                    pred_kpts=self.pred_keypoints,
                    target_kpts=target_kpts,
                    target_visibilities=target_visibilities,
                    seg_polygons=seg_polygons,
                )
            else:
                oks_value = -1
        else:
            oks_value = -1
        print(oks_value)

        raw_image = plot_connections(
            self.image.copy(),
            self.pred_keypoints,
            self.pred_scores,
            self.limbs,
            thr=0.05,
        )

        final_plots = []
        num_stages = 2
        for i in range(num_stages):
            kpts_heatmaps_plots = plot_heatmaps(
                self.input_image,
                self.pred_kpts_heatmaps[..., i],
                clip_0_1=True,
                minmax=False,
            )
            tags_heatmaps_plots = plot_heatmaps(
                self.input_image,
                self.pred_tags_heatmaps[..., i],
                clip_0_1=False,
                minmax=True,
            )
            kpts_grid = make_grid(kpts_heatmaps_plots, nrows=2, pad=5)
            tags_grid = make_grid(tags_heatmaps_plots, nrows=2, pad=5)
            final_plots.extend([kpts_grid, tags_grid])

        final_plot = np.concatenate(final_plots, axis=0)
        final_plot = cv2.resize(final_plot, dsize=(0, 0), fx=0.4, fy=0.4)
        return final_plot, raw_image
