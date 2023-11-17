from torch import nn, Tensor
from src.base.loss import BaseLoss, WeightedLoss


class KeypointsLoss(BaseLoss):
    def __init__(self):
        loss_criterion = nn.MSELoss()
        loss_fn = WeightedLoss(loss_criterion, weight=1)
        super().__init__(loss_fn=loss_fn)

    def calculate_loss(
        self,
        stages_pred_heatmaps: list[Tensor],
        stages_target_heatmaps: list[Tensor],
        target_weights: Tensor,
    ) -> Tensor:
        loss = 0
        target_masks = target_weights > 0
        for i in range(len(stages_pred_heatmaps)):
            pred_hm = stages_pred_heatmaps[i]
            target_hm = stages_target_heatmaps[i]
            stage_loss = self.loss_fn(pred_hm[target_masks], target_hm[target_masks])
            loss += stage_loss
        return loss
