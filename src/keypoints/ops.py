import torch
from torch import Tensor


def get_kpts_coords(heatmaps: Tensor) -> Tensor:
    """get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, "Score maps should be 4-dim"
    batch_size, num_kpts, h, w = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_kpts, -1)
    coords = torch.argmax(flat_heatmaps, 2)
    coords = coords.view(batch_size, num_kpts, 1) + 1
    coords = coords.repeat(1, 1, 2).float()
    coords[:, :, 0] = (coords[:, :, 0] - 1) % w + 1
    coords[:, :, 1] = torch.floor((coords[:, :, 1] - 1) / w) + 1
    # coords are in [x, y] order
    return coords
