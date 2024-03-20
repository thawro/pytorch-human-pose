import numpy as np

_head_coords = tuple[int, int, int, int]


def object_PCKh(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    head_xyxy: _head_coords,
    alpha: float = 0.5,
) -> float:
    if target_vis.sum() <= 0:
        return -1
    xmin, ymin, xmax, ymax = head_xyxy
    head_size = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    if head_size == 0:
        return -1
    norm_pred_kpts = pred_kpts / head_size
    norm_target_kpts = target_kpts / head_size
    sqared_diff = (norm_pred_kpts - norm_target_kpts) ** 2
    distances = sqared_diff.sum(-1) ** 0.5
    kpts_vis = target_vis > 0
    # pckh = (distances < alpha).sum().item()
    # pckh[~target_mask] = -1
    pckh = (distances < alpha) * 1
    pckh = pckh[kpts_vis]
    return pckh.mean()


def image_PCKh(
    alpha: float,
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    head_coords: list[_head_coords],
) -> float:
    num_obj = len(head_coords)
    pckhs = []
    for j in range(num_obj):
        pckh = object_PCKh(
            pred_kpts[j],
            target_kpts[j],
            target_vis[j],
            head_coords[j],
            alpha,
        )
        pckhs.append(pckh)
    pckhs = np.array(pckhs)
    valid_pckh_mask = pckhs != -1
    if valid_pckh_mask.sum() > 0:
        return pckhs[valid_pckh_mask].mean()
    return -1
