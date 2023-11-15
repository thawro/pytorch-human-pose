from torch import nn, Tensor
from .base import BaseLoss, WeightedLoss
import torchvision.transforms.functional as F


class KeypointsLoss(BaseLoss):
    def __init__(self):
        loss_criterion = nn.MSELoss()
        loss_fn = WeightedLoss(loss_criterion, weight=1)
        super().__init__(loss_fn=loss_fn)

    def calculate_loss(
        self,
        stages_pred_heatmaps: list[Tensor],
        target_heatmaps: Tensor,
        target_weights: Tensor,
    ) -> Tensor:
        loss = 0
        target_masks = target_weights > 0
        sized_target_heatmaps = {}
        target_h, target_w = target_heatmaps.shape[-2:]
        for stage_pred_heatmaps in stages_pred_heatmaps:
            pred_h, pred_w = stage_pred_heatmaps.shape[-2:]
            size = (pred_h, pred_w)
            if pred_h != target_h and pred_w != target_w:
                if size not in sized_target_heatmaps:
                    resized_target_heatmaps = F.resize(
                        target_heatmaps, size=[pred_h, pred_w]
                    )
                    sized_target_heatmaps[size] = resized_target_heatmaps
                else:
                    resized_target_heatmaps = sized_target_heatmaps[size]
            else:
                resized_target_heatmaps = target_heatmaps
                sized_target_heatmaps[size] = resized_target_heatmaps

            stage_loss = self.loss_fn(
                stage_pred_heatmaps[target_masks], resized_target_heatmaps[target_masks]
            )
            print(
                round(stage_loss.item(), 4),
                ": ",
                round(stage_pred_heatmaps.mean().item(), 4),
                round(stage_pred_heatmaps.min().item(), 4),
                round(stage_pred_heatmaps.max().item(), 4),
                "  <->  ",
                round(resized_target_heatmaps.mean().item(), 4),
                round(resized_target_heatmaps.min().item(), 4),
                round(resized_target_heatmaps.max().item(), 4),
            )
            loss += stage_loss
        print()
        return loss
