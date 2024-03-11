from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from src.base.results import BaseResult
from src.base.transforms.utils import affine_transform, get_affine_transform
from src.keypoints.grouping import MPPEHeatmapParser
from src.keypoints.metrics import OKS
from src.keypoints.transforms import KeypointsTransform
from src.keypoints.visualization import (
    plot_connections,
    plot_grouped_ae_tags,
    plot_heatmaps,
)
from src.utils.image import make_grid, match_size_to_src, stack_horizontally


def match_preds_to_targets(
    pred_joints: np.ndarray,
    pred_scores: np.ndarray,
    target_kpts: np.ndarray,
    target_visibilities: np.ndarray,
) -> list[int]:
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
            match_val = 1 / (((p_kpts[..., :2] - t_kpts[..., :2])[mask]) ** 2).sum(-1).mean()
            if match_val > target_matches_vals[target_idx]:
                target_matches_vals[target_idx] = match_val
                target_matches_idx[target_idx] = pred_idx
                matched_idxs.append(target_idx)
    return target_matches_idx


class BaseKeypointsResult(BaseResult):
    @classmethod
    def match_heatmaps_size(cls, heatmaps: list[Tensor]) -> list[Tensor]:
        h, w = heatmaps[-1].shape[-2:]
        heatmaps = [
            torch.nn.functional.interpolate(hm, size=[h, w], mode="bilinear", align_corners=False)
            for hm in heatmaps[:-1]
        ] + [heatmaps[-1]]
        return heatmaps

    @classmethod
    def resize_heatmaps_list(cls, heatmaps: list[Tensor], h: int, w: int) -> list[Tensor]:
        return [
            torch.nn.functional.interpolate(hm, size=[h, w], mode="bilinear", align_corners=False)
            for hm in heatmaps
        ]

    @classmethod
    def resize_heatmaps(cls, heatmaps: Tensor, h: int, w: int) -> Tensor:
        return torch.nn.functional.interpolate(
            heatmaps, size=[h, w], mode="bilinear", align_corners=False
        )


class KeypointsResult(BaseKeypointsResult):
    kpts_heatmaps: np.ndarray
    tags_heatmaps: np.ndarray

    def __init__(
        self,
        model_input_image: Tensor,
        kpts_heatmaps: list[Tensor],
        tags_heatmaps: Tensor,
        limbs: list[tuple[int, int]],
        max_num_people: int = 30,
        det_thr: float = 0.05,
        tag_thr: float = 0.5,
    ):
        self.model_input_image = KeypointsTransform.inverse_transform(model_input_image)
        self._kpts_heatmaps = kpts_heatmaps
        self._tags_heatmaps = tags_heatmaps
        self.num_kpts = kpts_heatmaps[0].shape[1]
        self.limbs = limbs
        self.max_num_people = max_num_people
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.hm_parser = MPPEHeatmapParser(self.num_kpts, max_num_people, det_thr, tag_thr)

    def set_preds(self):
        img_h, img_w = self.model_input_image.shape[:2]

        kpts_heatmaps = self.match_heatmaps_size(self._kpts_heatmaps)

        kpts_heatmaps = torch.stack(kpts_heatmaps, dim=1)  # stack the stages dim
        avg_kpts_heatmaps = kpts_heatmaps.mean(dim=1)

        resized_avg_kpts_heatmaps = self.resize_heatmaps(avg_kpts_heatmaps, img_h, img_w)

        resized_tags_heatmaps = self.resize_heatmaps(self._tags_heatmaps, img_h, img_w)
        resized_tags_heatmaps = resized_tags_heatmaps.unsqueeze(-1)  # add embedding dim as last dim

        # remove batch dim
        resized_tags_heatmaps = resized_tags_heatmaps[0]
        resized_avg_kpts_heatmaps = resized_avg_kpts_heatmaps[0]
        kpts_heatmaps = kpts_heatmaps[0]

        grouped_joints, obj_scores = self.hm_parser.parse(
            resized_avg_kpts_heatmaps, resized_tags_heatmaps, adjust=True, refine=True
        )

        self.kpts_coords = grouped_joints[..., :2]
        self.kpts_scores = grouped_joints[..., 2]
        self.kpts_tags = grouped_joints[..., 3:]
        self.obj_scores = obj_scores

        resized_kpts_heatmaps = self.resize_heatmaps(kpts_heatmaps, img_h, img_w)
        # num_stages, num_kpts, h, w -> num_kpts, h, w, num_stages
        self.kpts_heatmaps = resized_kpts_heatmaps.permute(1, 2, 3, 0).cpu().numpy()
        self.tags_heatmaps = resized_tags_heatmaps.cpu().numpy()

    def plot(self) -> dict[str, np.ndarray]:
        connections_plot = plot_connections(
            self.model_input_image.copy(),
            self.kpts_coords,
            self.kpts_scores,
            self.limbs,
            thr=self.det_thr,
        )
        stages_hms_plots = []
        num_stages = self.kpts_heatmaps.shape[-1]
        tags_embedding_dim = self.tags_heatmaps.shape[-1]
        for i in range(num_stages):
            stage_hms_plots = []
            kpts_hms_plot = plot_heatmaps(
                self.model_input_image, self.kpts_heatmaps[..., i], clip_0_1=True, minmax=False
            )
            kpts_hms_plot = make_grid(kpts_hms_plot, nrows=1, pad=5)
            stage_hms_plots.append(kpts_hms_plot)
            if i < tags_embedding_dim:
                tags_heatmaps_plots = plot_heatmaps(
                    self.model_input_image, self.tags_heatmaps[..., i], clip_0_1=False, minmax=True
                )
                tags_grid = make_grid(tags_heatmaps_plots, nrows=1, pad=5)
                stage_hms_plots.append(tags_grid)
            stages_hms_plots.extend(stage_hms_plots)
        stages_hms_plot = np.concatenate(stages_hms_plots, axis=0)
        stages_hms_plot = cv2.resize(stages_hms_plot, dsize=(0, 0), fx=0.4, fy=0.4)
        stages_hms_plot = stack_horizontally([connections_plot, stages_hms_plot])
        return {"heatmaps": stages_hms_plot}


def transform_coords(
    kpts_coords: np.ndarray,
    center: tuple[int, int],
    scale: tuple[float, float],
    output_size: tuple[int, int],
) -> np.ndarray:
    num_obj = kpts_coords.shape[0]
    transformed_kpts_coords = kpts_coords.copy()
    transform_matrix = get_affine_transform(center, scale, 0, output_size, inverse=True)
    for i in range(num_obj):
        transformed_kpts_coords[i, :2] = affine_transform(
            kpts_coords[i, :2].tolist(), transform_matrix
        )
    return transformed_kpts_coords


@dataclass
class InferenceKeypointsResult(BaseKeypointsResult):
    raw_image: np.ndarray
    annot: list[dict] | None
    model_input_image: np.ndarray
    kpts_heatmaps: np.ndarray
    tags_heatmaps: np.ndarray
    kpts_coords: np.ndarray
    kpts_scores: np.ndarray
    kpts_tags: np.ndarray
    obj_scores: np.ndarray
    limbs: list[tuple[int, int]]
    det_thr: float
    tag_thr: float

    @classmethod
    def get_final_kpts_coords(
        cls,
        kpts_coords: np.ndarray,
        center: tuple[int, int],
        scale: tuple[float, float],
        hm_size: tuple[int, int],
    ) -> np.ndarray:
        final_kpts_coords = []
        for person_kpts_coords in kpts_coords:
            person_kpts_coords = transform_coords(person_kpts_coords, center, scale, hm_size)
            final_kpts_coords.append(person_kpts_coords)
        return np.stack(final_kpts_coords)

    @classmethod
    def from_preds(
        cls,
        raw_image: np.ndarray,
        annot: list[dict] | None,
        model_input_image: Tensor,
        kpts_heatmaps: list[Tensor],
        tags_heatmaps: list[Tensor],
        limbs: list[tuple[int, int]],
        scale: tuple[float, float],
        center: tuple[int, int],
        det_thr: float = 0.05,
        tag_thr: float = 0.5,
        max_num_people: int = 30,
    ) -> "InferenceKeypointsResult":
        model_input_image_npy = KeypointsTransform.inverse_transform(model_input_image)
        img_h, img_w = model_input_image_npy.shape[:2]
        num_kpts = tags_heatmaps[0].shape[1]
        parser = MPPEHeatmapParser(
            num_kpts, max_num_people=max_num_people, det_thr=det_thr, tag_thr=tag_thr
        )

        kpts_heatmaps = cls.match_heatmaps_size(kpts_heatmaps)
        avg_kpts_heatmaps = torch.stack(kpts_heatmaps).mean(dim=0)
        resized_avg_kpts_heatmaps = cls.resize_heatmaps(avg_kpts_heatmaps, img_h, img_w)

        resized_tags_heatmaps = cls.resize_heatmaps_list(tags_heatmaps, img_h, img_w)
        resized_tags_heatmaps = torch.stack(resized_tags_heatmaps, dim=4)

        # remove batch dim
        resized_avg_kpts_heatmaps = resized_avg_kpts_heatmaps[0]
        resized_tags_heatmaps = resized_tags_heatmaps[0]

        grouped_joints, obj_scores = parser.parse(
            resized_avg_kpts_heatmaps, resized_tags_heatmaps, adjust=True, refine=True
        )

        kpts_coords = grouped_joints[..., :2]
        kpts_scores = grouped_joints[..., 2]
        kpts_tags = grouped_joints[..., 3:]

        kpts_coords = cls.get_final_kpts_coords(kpts_coords, center, scale, (img_w, img_h))

        kpts_heatmaps_npy = resized_avg_kpts_heatmaps.cpu().numpy()
        # only first tag embedding (for visualization purposes)
        tags_heatmaps_npy = resized_tags_heatmaps.cpu().numpy()[..., 0]

        return cls(
            raw_image=raw_image,
            annot=annot,
            model_input_image=model_input_image_npy,
            kpts_heatmaps=kpts_heatmaps_npy,
            tags_heatmaps=tags_heatmaps_npy,
            kpts_coords=kpts_coords,
            kpts_scores=kpts_scores,
            kpts_tags=kpts_tags,
            obj_scores=obj_scores,
            limbs=limbs,
            det_thr=det_thr,
            tag_thr=tag_thr,
        )

    def calculate_OKS(self) -> float:
        assert (
            self.annot is not None
        ), "Matching preds to targets is possible only when annotation is set"
        seg_polygons = []
        joints = []
        for obj in self.annot:
            obj_joints = np.array(obj["keypoints"]).reshape([-1, 3])
            is_any_kpts_visible = np.any(obj_joints[:, 2] > 0)
            if is_any_kpts_visible:
                joints.append(obj_joints)
                seg_polygons.append(obj["segmentation"])

        if len(joints) > 0:
            joints = np.stack(joints)
            target_kpts_coords = joints[..., :2]
            target_kpts_vis = joints[..., 2]
            target_matches_idx = match_preds_to_targets(
                self.kpts_coords,
                self.obj_scores,
                target_kpts_coords,
                target_kpts_vis,
            )
            if -1 not in target_matches_idx:
                self.kpts_coords = self.kpts_coords[target_matches_idx]
                self.kpts_scores = self.kpts_scores[target_matches_idx]
                self.obj_scores = self.obj_scores[target_matches_idx]
            oks = OKS()
            oks_value = oks.image_eval(
                pred_kpts=self.kpts_coords,
                target_kpts=target_kpts_coords,
                target_vis=target_kpts_vis,
                seg_polygons=seg_polygons,
            )
        return oks_value

    def plot(self) -> dict[str, np.ndarray]:
        if self.annot is not None:
            oks_value = self.calculate_OKS()
        else:
            oks_value = -1
        print(f"OKS: {oks_value:.2f}")

        connections_plot = plot_connections(
            self.raw_image.copy(),
            self.kpts_coords,
            self.kpts_scores,
            self.limbs,
            thr=self.det_thr,
        )

        kpts_hms_plot = plot_heatmaps(
            self.model_input_image, self.kpts_heatmaps, clip_0_1=False, minmax=True
        )
        kpts_hms_plot = make_grid(kpts_hms_plot, nrows=2, pad=5)

        tags_hms_plots = plot_heatmaps(
            self.model_input_image, self.tags_heatmaps, clip_0_1=False, minmax=True
        )
        tags_hms_plots = make_grid(tags_hms_plots, nrows=2, pad=5)

        hms_plot = np.concatenate([kpts_hms_plot, tags_hms_plots], axis=0)
        hms_plot = cv2.resize(hms_plot, dsize=(0, 0), fx=0.6, fy=0.6)

        ae_tags_plot = plot_grouped_ae_tags(self.kpts_tags)
        # ae_tags_plot = match_size_to_src(connections_plot, [ae_tags_plot], mode="height")[0]
        _connections_plot = match_size_to_src(ae_tags_plot, [connections_plot], mode="height")[0]

        ae_plot = stack_horizontally([_connections_plot, ae_tags_plot])
        return {
            "heatmaps": hms_plot,
            "connections": connections_plot,
            "associative_embedding": ae_plot,
        }
