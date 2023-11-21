from src.base.metrics import BaseMetrics
from torch import Tensor
import torch
from src.keypoints.ops import get_kpts_coords


class PercentageCorrectKeypoints:
    """Calculate accuracy according to PCK.

    PCK is calculated as percent of keypoints which are predicted in the range equal to
    thr * 0.1 * size
    from the ground trouth keypoint
    where thr is by default 0.5 and size is height for y coordinate and width for x coordinate
    """

    def __init__(self, idxs: list[int], thr: float = 0.5):
        self.idxs = idxs
        self.thr = thr

    def __call__(self, preds_heatmaps: Tensor, targets_heatmaps: Tensor) -> Tensor:
        batch_size, num_joints, h, w = preds_heatmaps.shape
        idxs = self.idxs
        if idxs is None:
            idxs = list(range(num_joints))

        # mask for cases when target has no annotation
        target_mask = targets_heatmaps.sum(dim=(2, 3)) > 0

        pred_coords = get_kpts_coords(preds_heatmaps)
        target_coords = get_kpts_coords(targets_heatmaps)

        # Using 1/10 of height and width as distance to gt kpt
        norm = torch.ones(batch_size, 2) * torch.tensor([w, h]) / 10
        norm = norm.unsqueeze(1).to(pred_coords.device)
        normed_pred_coords = pred_coords / norm
        normed_target_coords = target_coords / norm

        sqared_diff = (normed_pred_coords - normed_target_coords) ** 2
        distances = torch.sum(sqared_diff, dim=-1) ** (1 / 2)
        distances[~target_mask] = -1

        acc = torch.zeros(num_joints)
        for i in range(num_joints):
            kpt_dist = distances[:, idxs[i]]
            kpt_dist = kpt_dist[kpt_dist != -1]
            if len(kpt_dist) > 0:
                kpt_acc = 1.0 * (kpt_dist < self.thr).sum().item() / len(kpt_dist)
            else:
                kpt_acc = -1
            acc[i] = kpt_acc

        return acc


class KeypointsMetrics(BaseMetrics):
    def __init__(self):
        self.pck = PercentageCorrectKeypoints(idxs=None, thr=0.5)

    def calculate_metrics(
        self, preds_heatmaps: Tensor, targets_heatmaps: Tensor
    ) -> dict[str, float]:
        pck = self.pck(preds_heatmaps, targets_heatmaps)
        return {"PCK": pck.mean().item()}
