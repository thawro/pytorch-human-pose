from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from src.base.transforms.utils import affine_transform, get_affine_transform
from src.keypoints.grouping import MPPEHeatmapParser
from src.keypoints.metrics import OKS
from src.keypoints.transforms import KeypointsTransform
from src.keypoints.visualization import plot_connections, plot_heatmaps
from src.utils.image import make_grid


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


class KeypointsResult:
    kpts_heatmaps: np.ndarray
    tags_heatmaps: np.ndarray

    def __init__(
        self,
        image: Tensor,
        kpts_heatmaps: list[Tensor],
        tags_heatmaps: Tensor,
        limbs: list[tuple[int, int]],
        max_num_people: int = 30,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
    ):
        self.image = KeypointsTransform.inverse_transform(image)
        self._kpts_heatmaps = kpts_heatmaps
        self._tags_heatmaps = tags_heatmaps
        self.num_kpts = kpts_heatmaps[0].shape[0]
        self.limbs = limbs
        self.max_num_people = max_num_people
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.hm_parser = MPPEHeatmapParser(self.num_kpts, max_num_people, det_thr, tag_thr)

    def set_preds(self):
        img_h, img_w = self.image.shape[:2]
        num_stages = len(self.kpts_heatmaps)

        kpts_heatmaps = [
            torch.nn.functional.interpolate(
                self._kpts_heatmaps[i].unsqueeze(0),
                size=[img_h, img_w],
                mode="bilinear",
                align_corners=False,
            )
            for i in range(num_stages)
        ]

        tags_heatmaps = torch.nn.functional.interpolate(
            self._tags_heatmaps.unsqueeze(0),
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(-1)

        kpts_heatmaps = torch.stack(kpts_heatmaps, dim=-1)

        avg_kpts_heatmaps = kpts_heatmaps.mean(-1)

        avg_kpts_heatmaps = torch.nn.functional.interpolate(
            avg_kpts_heatmaps,
            size=[img_h, img_w],
            mode="bilinear",
            align_corners=False,
        )
        grouped_joints, obj_scores = self.hm_parser.parse(
            avg_kpts_heatmaps, tags_heatmaps, adjust=True, refine=True
        )

        kpts_coords = grouped_joints[..., :2]
        kpts_scores = grouped_joints[..., 2]

        self.kpts_coords = kpts_coords
        self.kpts_scores = kpts_scores
        self.obj_scores = obj_scores

        self.kpts_heatmaps = kpts_heatmaps.cpu().numpy()[0]
        self.tags_heatmaps = tags_heatmaps.cpu().numpy()[0]

    def plot(self) -> np.ndarray:
        image_limbs = plot_connections(
            self.image.copy(),
            self.kpts_coords,
            self.kpts_scores,
            self.limbs,
            thr=self.det_thr,
        )

        final_plots = []
        num_stages = 2
        for i in range(num_stages):
            kpts_heatmaps_plots = plot_heatmaps(
                self.image,
                self.kpts_heatmaps[..., i],
                clip_0_1=True,
                minmax=False,
            )
            kpts_heatmaps_plots.insert(0, image_limbs)
            kpts_grid = make_grid(kpts_heatmaps_plots, nrows=1, pad=5)
            _plots = [kpts_grid]
            if i < self.tags_heatmaps.shape[-1]:
                tags_heatmaps_plots = plot_heatmaps(
                    self.image,
                    self.tags_heatmaps[..., i],
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
class InferenceKeypointsResult:
    annot: np.ndarray | None
    input_image: np.ndarray
    image: np.ndarray
    kpts_heatmaps: np.ndarray
    tags_heatmaps: np.ndarray
    kpts_coords: np.ndarray
    kpts_scores: np.ndarray
    obj_scores: np.ndarray
    limbs: list[tuple[int, int]]
    det_thr: float
    tag_thr: float

    @classmethod
    def resize_stages_kpts_heatmaps(cls, kpts_heatmaps: list[Tensor]) -> list[Tensor]:
        h, w = kpts_heatmaps[-1].shape[-2:]
        kpts_heatmaps = [
            torch.nn.functional.interpolate(hm, size=[h, w], mode="bilinear", align_corners=False)
            for hm in kpts_heatmaps[:-1]
        ] + [kpts_heatmaps[-1]]
        return kpts_heatmaps

    @classmethod
    def average_kpts_heatmaps(cls, kpts_heatmaps: list[Tensor]) -> Tensor:
        return torch.stack(kpts_heatmaps, dim=-1).mean(dim=-1)

    @classmethod
    def resize_kpts_heatmaps(cls, kpts_heatmaps: Tensor, h: int, w: int) -> Tensor:
        return torch.nn.functional.interpolate(
            kpts_heatmaps, size=[h, w], mode="bilinear", align_corners=False
        )

    @classmethod
    def resize_tags_heatmaps(cls, tags_heatmaps: list[Tensor], h: int, w: int) -> Tensor:
        tags_heatmaps = [
            torch.nn.functional.interpolate(hms, size=[h, w], mode="bilinear", align_corners=False)
            for hms in tags_heatmaps
        ]
        return torch.stack(tags_heatmaps, dim=4)

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
        input_image: Tensor,
        image: np.ndarray,
        scale: tuple[float, float],
        center: tuple[int, int],
        kpts_heatmaps: list[Tensor],
        tags_heatmaps: list[Tensor],
        limbs: list[tuple[int, int]],
        max_num_people: int = 20,
        det_thr: float = 0.1,
        tag_thr: float = 1,
        annot: list[dict] | None = None,
    ) -> "InferenceKeypointsResult":
        input_image_npy = KeypointsTransform.inverse_transform(input_image)
        img_h, img_w = input_image_npy.shape[:2]
        num_kpts = tags_heatmaps[0].shape[1]
        parser = MPPEHeatmapParser(
            num_kpts, max_num_people=max_num_people, det_thr=det_thr, tag_thr=tag_thr
        )

        kpts_heatmaps = cls.resize_stages_kpts_heatmaps(kpts_heatmaps)
        kpts_heatmaps = cls.average_kpts_heatmaps(kpts_heatmaps)
        resized_kpts_heatmaps = cls.resize_kpts_heatmaps(kpts_heatmaps, img_h, img_w)

        resized_tags_heatmaps = cls.resize_tags_heatmaps(tags_heatmaps, img_h, img_w)

        grouped_joints, obj_scores = parser.parse(
            resized_kpts_heatmaps, resized_tags_heatmaps, adjust=True, refine=True
        )

        kpts_coords = grouped_joints[..., :2]
        kpts_scores = grouped_joints[..., 2]

        kpts_coords = cls.get_final_kpts_coords(kpts_coords, center, scale, (img_w, img_h))

        kpts_heatmaps_npy = resized_kpts_heatmaps.cpu().numpy()[0]
        # only first tag embedding (for visualization purposes)
        tags_heatmaps_npy = resized_tags_heatmaps.cpu().numpy()[0, ..., 0]

        return cls(
            annot,
            input_image_npy,
            image,
            kpts_heatmaps_npy,
            tags_heatmaps_npy,
            kpts_coords,
            kpts_scores,
            obj_scores,
            limbs,
            det_thr,
            tag_thr,
        )

    def calculate_OKS(self) -> float:
        assert (
            self.annot is not None
        ), "Matching preds to targets is possible only when annotation is set"
        objects = self.annot["objects"]
        target_kpts = []
        target_visibilities = []
        seg_polygons = []
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
                self.kpts_coords,
                self.obj_scores,
                target_kpts,
                target_visibilities,
            )
            if -1 not in target_matches_idx:
                self.kpts_coords = self.kpts_coords[target_matches_idx]
                self.kpts_scores = self.kpts_scores[target_matches_idx]
                self.obj_scores = self.obj_scores[target_matches_idx]
            oks = OKS()
            oks_value = oks.image_eval(
                pred_kpts=self.kpts_coords,
                target_kpts=target_kpts,
                target_visibilities=target_visibilities,
                seg_polygons=seg_polygons,
            )
        return oks_value

    def plot(self) -> tuple[np.ndarray, np.ndarray]:
        if self.annot is not None:
            oks_value = self.calculate_OKS()
        else:
            oks_value = -1
        print(oks_value)

        raw_image = plot_connections(
            self.image.copy(),
            self.kpts_coords,
            self.kpts_scores,
            self.limbs,
            thr=self.det_thr / 2,
        )

        final_plots = []

        kpts_heatmaps_plots = plot_heatmaps(
            self.input_image,
            self.kpts_heatmaps,
            clip_0_1=False,
            minmax=True,
        )
        kpts_grid = make_grid(kpts_heatmaps_plots, nrows=2, pad=5)

        tags_heatmaps_plots = plot_heatmaps(
            self.input_image,
            self.tags_heatmaps,
            clip_0_1=False,
            minmax=True,
        )
        tags_grid = make_grid(tags_heatmaps_plots, nrows=2, pad=5)
        final_plots.extend([kpts_grid, tags_grid])

        final_plot = np.concatenate(final_plots, axis=0)
        f_ = 0.6
        final_plot = cv2.resize(final_plot, dsize=(0, 0), fx=f_, fy=f_)
        return final_plot, raw_image
