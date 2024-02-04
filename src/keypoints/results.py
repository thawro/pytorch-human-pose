from dataclasses import dataclass, field
import numpy as np
from torch import Tensor
from src.keypoints.grouping import SPPEHeatmapParser, MPPEHeatmapParser

from src.keypoints.metrics import OKS, PCKh, object_PCKh, object_OKS, EvaluationMetric
from typing import Callable
import torchvision.transforms.functional as F
import torch
from src.keypoints.visualization import plot_heatmaps, plot_connections
import cv2
from src.utils.image import make_grid


def match_preds_to_targets(
    pred_joints: np.ndarray,
    pred_scores: np.ndarray,
    target_kpts: np.ndarray,
    target_visibilities: np.ndarray,
) -> list[int]:
    # pred_kpts shape: [num_obj_pred, num_kpts, 3]
    # 3 for: x, y, score
    # target_kpts shape: [num_obj_target, num_kpts, 2]
    # 2 for: x, y
    # COCO: area segmentation, MPII: head_xyxy coords
    num_target_obj, num_kpts = target_kpts.shape[:2]
    sorted_idxs = np.argsort(pred_scores, kind="mergesort")
    target_matches_idx = [-1 for _ in range(num_target_obj)]
    target_matches_vals = [-np.inf for _ in range(num_target_obj)]
    matched_idxs = []
    for pred_idx in sorted_idxs:
        p_kpts = pred_joints[pred_idx]
        for target_idx in range(len(target_kpts)):
            t_kpts = target_kpts[target_idx]
            t_vis = target_visibilities[target_idx]
            mask = t_vis > 0
            # TODO: change match_fn to do normal dist between points
            match_val = (
                1 / (((p_kpts[..., :2] - t_kpts[..., :2])[mask]) ** 2).sum(-1).mean()
            )
            # if target_idx in matched_idxs:
            #     continue
            if match_val > target_matches_vals[target_idx]:
                target_matches_vals[target_idx] = match_val
                target_matches_idx[target_idx] = pred_idx
                matched_idxs.append(target_idx)
    return target_matches_idx


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


class MPPEKeypointsResult:
    annot: np.ndarray | None
    image: np.ndarray
    pred_kpts_heatmaps: np.ndarray
    pred_tags_heatmaps: np.ndarray
    pred_keypoints: np.ndarray
    pred_scores: np.ndarray
    limbs: list[tuple[int, int]]

    def __init__(
        self,
        image: np.ndarray,
        stages_pred_kpts_heatmaps: list[Tensor],
        stages_pred_tags_heatmaps: list[Tensor],
        limbs: list[tuple[int, int]],
        max_num_people: int = 30,
        det_thr: float = 0.1,
        tag_thr: float = 0.1,
    ):
        self.image = image
        self.stages_pred_kpts_heatmaps = stages_pred_kpts_heatmaps
        self.stages_pred_tags_heatmaps = stages_pred_tags_heatmaps
        self.num_kpts = stages_pred_kpts_heatmaps[0].shape[0]
        self.limbs = limbs
        self.max_num_people = max_num_people
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.hm_parser = MPPEHeatmapParser(
            self.num_kpts, max_num_people, det_thr, tag_thr
        )

    def set_preds(self):
        if hasattr(self, "pred_keypoints"):
            print("Preds already set. Returning")
            return
        img_h, img_w = self.image.shape[:2]
        h, w = self.stages_pred_kpts_heatmaps[-1].shape[-2:]
        num_stages = len(self.stages_pred_kpts_heatmaps)

        stages_pred_kpts_heatmaps = [
            torch.nn.functional.interpolate(
                self.stages_pred_kpts_heatmaps[i].unsqueeze(0),
                size=[img_h, img_w],
                mode="bilinear",
                align_corners=False,
            )
            for i in range(num_stages)
        ]

        pred_tags_heatmaps = torch.nn.functional.interpolate(
            self.stages_pred_tags_heatmaps[0].unsqueeze(0),
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(-1)

        pred_kpts_heatmaps = torch.stack(stages_pred_kpts_heatmaps, dim=-1)

        kpts_hms_to_parse = pred_kpts_heatmaps.mean(-1)

        kpts_hms_to_parse = torch.nn.functional.interpolate(
            kpts_hms_to_parse,
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        )
        joints, pred_obj_scores = self.hm_parser.parse(
            kpts_hms_to_parse, pred_tags_heatmaps, adjust=True, refine=True
        )

        if len(joints[0]) > 0:
            joints = np.stack(joints, axis=0)
            pred_kpts_coords = joints[0][..., :2]
            pred_kpts_scores = joints[0][..., 2]
        else:
            pred_obj_scores = np.array([0])
            pred_kpts_coords = np.zeros((1, 17, 2))
            pred_kpts_scores = np.zeros((1, 17, 1))
            pred_kpts_coords[:, :, 0] = img_w // 2
            pred_kpts_coords[:, :, 1] = img_h // 2

        self.pred_keypoints = pred_kpts_coords
        self.pred_scores = pred_kpts_scores
        self.pred_obj_scores = pred_obj_scores

        pred_kpts_heatmaps = pred_kpts_heatmaps.cpu().numpy()
        pred_tags_heatmaps = pred_tags_heatmaps.cpu().numpy()

        self.pred_kpts_heatmaps = pred_kpts_heatmaps[0]
        self.pred_tags_heatmaps = pred_tags_heatmaps[0]

    def plot(self) -> np.ndarray:
        image_limbs = plot_connections(
            self.image.copy(),
            self.pred_keypoints,
            self.pred_scores,
            self.limbs,
            thr=self.det_thr,
        )

        final_plots = []
        num_stages = 2
        for i in range(num_stages):
            kpts_heatmaps_plots = plot_heatmaps(
                self.image,
                self.pred_kpts_heatmaps[..., i],
                clip_0_1=True,
                minmax=False,
            )
            kpts_heatmaps_plots.insert(0, image_limbs)
            kpts_grid = make_grid(kpts_heatmaps_plots, nrows=1, pad=5)
            _plots = [kpts_grid]
            if i < self.pred_tags_heatmaps.shape[-1]:
                tags_heatmaps_plots = plot_heatmaps(
                    self.image,
                    self.pred_tags_heatmaps[..., i],
                    clip_0_1=False,
                    minmax=True,
                )
                tags_heatmaps_plots.insert(0, image_limbs)
                tags_grid = make_grid(tags_heatmaps_plots, nrows=1, pad=5)
                _plots.append(tags_grid)
            final_plots.extend(_plots)

        final_plot = np.concatenate(final_plots, axis=0)
        final_plot = cv2.resize(final_plot, dsize=(0, 0), fx=0.4, fy=0.4)
        return final_plot


### Inference


@dataclass
class InferenceMPPEKeypointsResult:
    annot: np.ndarray | None
    model_input: np.ndarray
    image: np.ndarray
    pred_kpts_heatmaps: np.ndarray
    pred_tags_heatmaps: np.ndarray
    pred_keypoints: np.ndarray
    pred_scores: np.ndarray
    pred_obj_scores: np.ndarray
    limbs: list[tuple[int, int]]

    @classmethod
    def from_preds(
        cls,
        annot,
        model_input: np.ndarray,
        image: np.ndarray,
        scale: float,
        center: tuple[int, int],
        stages_pred_kpts_heatmaps: list[Tensor],
        stages_pred_tags_heatmaps: list[Tensor],
        get_final_preds,
        limbs: list[tuple[int, int]],
        max_num_people: int = 20,
        det_thr: float = 0.1,
        tag_thr: float = 1,
    ) -> "InferenceMPPEKeypointsResult":
        img_h, img_w = model_input.shape[:2]
        h, w = stages_pred_kpts_heatmaps[-1].shape[-2:]
        stages_pred_kpts_heatmaps = [
            torch.nn.functional.interpolate(
                hm,
                size=[h, w],
                mode="bilinear",
                align_corners=False,
            )
            for hm in stages_pred_kpts_heatmaps[:-1]
        ] + [stages_pred_kpts_heatmaps[-1]]
        pred_tags_heatmaps = torch.nn.functional.interpolate(
            stages_pred_tags_heatmaps[0],
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(-1)

        num_kpts = stages_pred_kpts_heatmaps[0].shape[1]

        pred_kpts_heatmaps = torch.stack(stages_pred_kpts_heatmaps, dim=-1).mean(dim=-1)

        batch_size, num_kpts = pred_kpts_heatmaps.shape[:2]
        parser = MPPEHeatmapParser(
            num_kpts, max_num_people=max_num_people, det_thr=det_thr, tag_thr=tag_thr
        )
        pred_kpts_heatmaps = torch.nn.functional.interpolate(
            pred_kpts_heatmaps,
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        )
        joints, pred_obj_scores = parser.parse(
            pred_kpts_heatmaps, pred_tags_heatmaps, adjust=True, refine=True
        )

        final_results = get_final_preds(joints, center, scale, [img_w, img_h])
        if len(final_results) > 0:
            final_results = np.stack(final_results, axis=0)
            pred_kpts_coords = final_results[..., :2]
            pred_kpts_scores = joints[..., 2]
        else:
            pred_obj_scores = np.array([0])
            pred_kpts_coords = np.zeros((1, 17, 2))
            pred_kpts_scores = np.zeros((1, 17, 1))
            pred_kpts_coords[:, :, 0] = img_w // 2
            pred_kpts_coords[:, :, 1] = img_h // 2

        return cls(
            annot,
            model_input,
            image,
            pred_kpts_heatmaps.cpu().numpy()[0],
            pred_tags_heatmaps.cpu().numpy()[0, ..., 0],
            pred_kpts_coords,
            pred_kpts_scores,
            pred_obj_scores,
            limbs,
        )

    def plot(self) -> tuple[np.ndarray, np.ndarray]:
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
                target_matches_idx = match_preds_to_targets(
                    self.pred_keypoints,
                    self.pred_obj_scores,
                    target_kpts,
                    target_visibilities,
                )
                if -1 not in target_matches_idx:
                    self.pred_keypoints = self.pred_keypoints[target_matches_idx]
                    self.pred_scores = self.pred_scores[target_matches_idx]
                    self.pred_obj_scores = self.pred_obj_scores[target_matches_idx]

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

        # raw_image = self.image.copy()
        raw_image = plot_connections(
            self.image.copy(),
            self.pred_keypoints,
            self.pred_scores,
            self.limbs,
            thr=0.05,
        )

        final_plots = []

        kpts_heatmaps_plots = plot_heatmaps(
            self.model_input,
            self.pred_kpts_heatmaps,
            clip_0_1=False,
            minmax=True,
        )
        kpts_grid = make_grid(kpts_heatmaps_plots, nrows=2, pad=5)

        tags_heatmaps_plots = plot_heatmaps(
            self.model_input,
            self.pred_tags_heatmaps,
            clip_0_1=False,
            minmax=True,
        )
        tags_grid = make_grid(tags_heatmaps_plots, nrows=2, pad=5)
        final_plots.extend([kpts_grid, tags_grid])

        final_plot = np.concatenate(final_plots, axis=0)
        f_ = 0.6
        final_plot = cv2.resize(final_plot, dsize=(0, 0), fx=f_, fy=f_)
        return final_plot, raw_image
