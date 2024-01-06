import numpy as np
import cv2
from abc import abstractmethod

_polygons = list[list[int]]
_head_coords = list[list[int]]  # one-element list (to match COCO polygons)

sigmas = [26, 25, 25, 35, 35, 79, 79, 72, 72, 62, 62, 107, 107, 87, 87, 89, 89]
sigmas = np.array(sigmas) / 1000
variances = (sigmas * 2) ** 2


def object_PCKh(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    head_xyxy: _head_coords,
    alpha: float = 0.5,
) -> float:
    xmin, ymin, xmax, ymax = head_xyxy[0]
    head_size = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    if head_size == 0:
        return -1
    norm_pred_kpts = pred_kpts / head_size
    norm_target_kpts = target_kpts / head_size
    sqared_diff = (norm_pred_kpts - norm_target_kpts) ** 2
    distances = sqared_diff.sum(-1) ** 0.5
    # both coords must be seen
    kpts_vis = np.array([x > 0 and y > 0 for x, y in target_kpts])
    # pckh = (distances < alpha).sum().item()
    # pckh[~target_mask] = -1
    if kpts_vis.sum() > 0:
        pckh = (distances < alpha) * 1
        pckh = pckh[kpts_vis]
        return pckh.mean()
    else:
        return -1


def object_OKS(
    pred_kpts: np.ndarray, target_kpts: np.ndarray, obj_polygons: _polygons
) -> float:
    # pred_kpts shape: [num_kpts, 2]
    # target_kpts shape: [num_kpts, 2]
    # 2 for: x, y
    num_kpts = len(target_kpts)

    kpts_vis = np.array([x > 0 or y > 0 for x, y in target_kpts[..., :2]])

    if kpts_vis.sum() > 0:
        area = sum(
            cv2.contourArea(np.array(poly).reshape(-1, 2)) for poly in obj_polygons
        )
        dist = ((pred_kpts - target_kpts) ** 2).sum(-1)
        e = dist / variances / (area + np.spacing(1)) / 2
        e = e[kpts_vis]
        return np.sum(np.exp(-e)) / num_kpts
    else:
        return -1


class EvaluationMetric:
    @abstractmethod
    def image_eval(
        self, pred_kpts: np.ndarray, target_kpts: np.ndarray, extra_coords: list
    ):
        raise NotImplementedError()

    def evaluate_results(
        self, pred_kpts: np.ndarray, target_kpts: np.ndarray, extra_coords: list
    ) -> float:
        batch_size = len(pred_kpts)
        metric_values = []
        for i in range(batch_size):
            metric_value = self.image_eval(
                pred_kpts[i],
                target_kpts[i],
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
        seg_polygons: list[_polygons],
    ) -> float:
        num_obj = len(seg_polygons)
        oks_values = []
        for j in range(num_obj):
            pred_kpt_coords = pred_kpts[j]
            target_kpt_coords = target_kpts[j]
            obj_polygons = seg_polygons[j]
            oks = object_OKS(pred_kpt_coords, target_kpt_coords, obj_polygons)
            oks_values.append(oks)
        oks_values = np.array(oks_values)

        valid_oks_mask = oks_values != -1
        if valid_oks_mask.sum() > 0:
            return oks_values[valid_oks_mask].mean()
        return -1

    def evaluate_results(
        self, pred_kpts: np.ndarray, target_kpts: np.ndarray, extra_coords: list
    ):
        oks = super().evaluate_results(pred_kpts, target_kpts, extra_coords)
        return {"OKS": oks}


class PCKh(EvaluationMetric):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def image_eval(
        self,
        pred_kpts: np.ndarray,
        target_kpts: np.ndarray,
        head_coords: list[list[list[int]]],
    ) -> float:
        num_obj = len(head_coords)
        pckhs = []
        for j in range(num_obj):
            pred_kpt_coords = pred_kpts[j]
            target_kpt_coords = target_kpts[j]
            head_xyxy = head_coords[j]
            pckh = object_PCKh(
                pred_kpt_coords, target_kpt_coords, head_xyxy, self.alpha
            )
            pckhs.append(pckh)
        pckhs = np.array(pckhs)
        valid_pckh_mask = pckhs != -1
        if valid_pckh_mask.sum() > 0:
            return pckhs[valid_pckh_mask].mean()
        return -1

    def evaluate_results(
        self, pred_kpts: np.ndarray, target_kpts: np.ndarray, extra_coords: list
    ):
        pckh = super().evaluate_results(pred_kpts, target_kpts, extra_coords)
        return {f"PCKh@{self.alpha}": pckh}
