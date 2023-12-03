import torch
from torch import Tensor
import numpy as np
import torchvision.transforms.functional as F


def get_sppe_kpts_coords(
    heatmaps: Tensor, return_scores: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Return keypoints coordinates for Single Person Pose Estimation (SPPE)"""
    if isinstance(heatmaps, np.ndarray):
        heatmaps = torch.from_numpy(heatmaps)
    if heatmaps.dim() == 3:
        heatmaps.unsqueeze_(0)
    assert heatmaps.dim() == 4, f"Score maps should be 4-dim (actual: {heatmaps.dim()})"
    batch_size, num_kpts, h, w = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_kpts, -1)
    coords = torch.argmax(flat_heatmaps, 2)
    coords = coords.view(batch_size, num_kpts, 1) + 1
    coords = coords.repeat(1, 1, 2).float()
    coords[:, :, 0] = (coords[:, :, 0] - 1) % w
    coords[:, :, 1] = torch.floor((coords[:, :, 1] - 1) / w)
    coords = coords.to(torch.int32).flip(-1)
    # coords are in [y, x] order

    if return_scores:
        scores = torch.zeros(batch_size, num_kpts)
        for i in range(batch_size):
            for j in range(num_kpts):
                y, x = coords[i, j].tolist()
                scores[i, j] = heatmaps[i, j][y, x]
        return coords, scores
    return coords


# TODO
def get_mppe_ae_kpts_coords(
    heatmaps: Tensor, tags: Tensor, return_scores: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Return keypoints coordinates for Multi Person Pose Estimation (MPPE) using tags (Associative Embeddings)"""
    if isinstance(heatmaps, np.ndarray):
        heatmaps = torch.from_numpy(heatmaps)
        tags = torch.from_numpy(tags)

    if heatmaps.dim() == 3:
        heatmaps.unsqueeze_(0)
        tags.unsqueeze_(0)

    assert (
        heatmaps.dim() == 4 and tags.dim() == 4
    ), f"Score maps should be 4-dim (actual: {heatmaps.dim(), tags.dim()})"

    batch_size, num_kpts, h, w = heatmaps.shape
    tags = F.resize(tags, [h, w])
    print(tags.shape, heatmaps.shape)
    exit()
    coords = None

    if return_scores:
        num_obj = None
        scores = torch.zeros(batch_size, num_obj, num_kpts)
        for i in range(batch_size):
            for j in range(num_kpts):
                y, x = coords[i, j].tolist()
                scores[i, j] = heatmaps[i, j][y, x]
        return coords, scores
    return coords
