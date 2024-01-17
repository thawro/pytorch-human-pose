import numpy as np
import cv2
from abc import abstractmethod

_polygons = list[list[int]]
_head_coords = list[list[int]]  # one-element list (to match COCO polygons)


k_i = [26, 25, 25, 35, 35, 79, 79, 72, 72, 62, 62, 107, 107, 87, 87, 89, 89]
k_i = np.array(k_i) / 1000
variances = (k_i * 2) ** 2


def object_PCKh(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_visibilities: np.ndarray,
    head_xyxy: _head_coords,
    alpha: float = 0.5,
) -> float:
    if target_visibilities.sum() <= 0:
        return -1
    xmin, ymin, xmax, ymax = head_xyxy[0]
    head_size = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    if head_size == 0:
        return -1
    norm_pred_kpts = pred_kpts / head_size
    norm_target_kpts = target_kpts / head_size
    sqared_diff = (norm_pred_kpts - norm_target_kpts) ** 2
    distances = sqared_diff.sum(-1) ** 0.5
    # both coords must be seen
    # kpts_vis = np.array([x > 0 and y > 0 for x, y in target_kpts])
    kpts_vis = target_visibilities > 0
    # pckh = (distances < alpha).sum().item()
    # pckh[~target_mask] = -1
    pckh = (distances < alpha) * 1
    pckh = pckh[kpts_vis]
    return pckh.mean()


def object_OKS(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_visibilities: np.ndarray,
    obj_polygons: _polygons,
) -> float:
    # pred_kpts shape: [num_kpts, 2]
    # target_kpts shape: [num_kpts, 2]
    # 2 for: x, y
    if target_visibilities.sum() <= 0:
        return -1

    kpts_vis = target_visibilities > 0
    area = sum(
        cv2.contourArea(np.array(poly).reshape(-1, 2).astype(np.int32))
        for poly in obj_polygons
    )
    area += np.spacing(1)

    dist = ((pred_kpts - target_kpts) ** 2).sum(-1)
    # dist is already squared (euclidean distance has square root)
    e = dist / (2 * variances * area)
    e = e[kpts_vis]
    num_vis_kpts = kpts_vis.sum()
    e = np.exp(-e)
    oks = np.sum(e) / num_vis_kpts
    return oks


class EvaluationMetric:
    @abstractmethod
    def image_eval(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        extra_coords: list,
    ):
        raise NotImplementedError()

    def evaluate_results(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        extra_coords: list,
    ) -> float:
        batch_size = len(pred_kpts)
        metric_values = []
        for i in range(batch_size):
            metric_value = self.image_eval(
                pred_kpts[i],
                target_kpts[i],
                target_visibilities[i],
                extra_coords[i],
            )
            metric_values.append(metric_value)
        metric_values = np.array(metric_values)

        valid_results = metric_values != -1
        if valid_results.sum() > 0:
            return metric_values[valid_results].mean().item()
        print("   ----->   ALL BATCH WRONG RESULTS")
        return -1


class OKS(EvaluationMetric):
    def image_eval(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        seg_polygons: list[_polygons],
    ) -> float:
        num_obj = len(seg_polygons)
        oks_values = []
        for j in range(num_obj):
            dist = ((pred_kpts[j] - target_kpts[j]) ** 2).sum(-1) ** 0.5
            oks = object_OKS(
                pred_kpts[j], target_kpts[j], target_visibilities[j], seg_polygons[j]
            )
            oks_values.append(oks)

        oks_values = np.array(oks_values).round(3)
        valid_oks_mask = oks_values != -1
        if valid_oks_mask.sum() > 0:
            oks_avg = oks_values[valid_oks_mask].mean()
            return oks_avg
        return -1

    def evaluate_results(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        extra_coords: list,
    ):
        oks = super().evaluate_results(
            pred_kpts, target_kpts, target_visibilities, extra_coords
        )
        return {"OKS": oks}


class PCKh(EvaluationMetric):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def image_eval(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        head_coords: list[list[list[int]]],
    ) -> float:
        num_obj = len(head_coords)
        pckhs = []
        for j in range(num_obj):
            pckh = object_PCKh(
                pred_kpts[j],
                target_kpts[j],
                target_visibilities[j],
                head_coords[j],
                self.alpha,
            )
            pckhs.append(pckh)
        pckhs = np.array(pckhs)
        valid_pckh_mask = pckhs != -1
        if valid_pckh_mask.sum() > 0:
            return pckhs[valid_pckh_mask].mean()
        return -1

    def evaluate_results(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        target_visibilities: np.ndarray,
        extra_coords: list,
    ):
        pckh = super().evaluate_results(
            pred_kpts, target_kpts, target_visibilities, extra_coords
        )
        return {f"PCKh@{self.alpha}": pckh}
