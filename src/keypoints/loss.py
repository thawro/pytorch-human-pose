from torch import nn, Tensor
from src.base.loss import BaseLoss, WeightedLoss
from torch.nn.modules.loss import _Loss
import torch


class HeatmapsLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(
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
            stage_loss = self.criterion(pred_hm[target_masks], target_hm[target_masks])
            loss += stage_loss
        return loss


class KeypointsLoss(BaseLoss):
    def __init__(self):
        super().__init__(WeightedLoss(HeatmapsLoss(), weight=1))

    def calculate_loss(
        self,
        stages_pred_heatmaps: list[Tensor],
        stages_target_heatmaps: list[Tensor],
        target_weights: Tensor,
    ) -> Tensor:
        return self.loss_fn(
            stages_pred_heatmaps, stages_target_heatmaps, target_weights
        )


class AEGroupingLoss(_Loss):
    def forward(self, pred_tags: list[Tensor], target_keypoints: list):
        pred_tags = pred_tags[-1]
        batch_size = len(target_keypoints)
        pull_loss = 0
        push_loss = 0
        for i in range(batch_size):
            tags = pred_tags[i]
            all_objs_kpts = target_keypoints[i]

            all_objs_pull_loss = 0  # intra object kpts
            all_objs_ref_tags = []  # ^h_n
            for j, obj_kpts in enumerate(all_objs_kpts):
                obj_kpts_tags = []
                for k, kpt in enumerate(obj_kpts):
                    x, y = kpt
                    # coords were wrt image size, now we need to parse it to tags size
                    x, y = x // 4, y // 4
                    if x > 0 and y > 0:
                        tag = tags[k][y, x]
                        obj_kpts_tags.append(tag)
                obj_kpts_tags = torch.Tensor(obj_kpts_tags)
                obj_ref_tag = obj_kpts_tags.mean()
                all_objs_ref_tags.append(obj_ref_tag)

                obj_pull_loss = ((obj_kpts_tags - obj_ref_tag) ** 2).sum()
                all_objs_pull_loss += obj_pull_loss

            num_obj = len(all_objs_ref_tags)
            all_objs_pull_loss /= num_obj

            if num_obj == 0:
                continue
            elif num_obj == 1:
                pull_loss += all_objs_pull_loss
            else:
                pull_loss += all_objs_pull_loss
                ref_tags = torch.stack(all_objs_ref_tags)

                size = (num_obj, num_obj)
                A = ref_tags.expand(*size)
                B = A.permute(1, 0)
                diff = A - B

                diff = torch.pow(diff, 2)
                _push_loss = torch.exp(-diff)
                _push_loss = torch.sum(_push_loss) - num_obj
                _pull_loss = _push_loss / ((num_obj - 1) * num_obj) * 0.5
                push_loss += _pull_loss
        return push_loss + pull_loss


class AEKeypointsLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.heatmaps_loss = WeightedLoss(HeatmapsLoss(), weight=1)
        self.tags_loss = WeightedLoss(AEGroupingLoss(), weight=1e-3)

    def calculate_loss(
        self,
        stages_pred_heatmaps: list[Tensor],
        stages_target_heatmaps: list[Tensor],
        stages_pred_tags: list[Tensor],
        target_weights: Tensor,
        target_keypoints: list,
    ) -> Tensor:
        heamtaps_loss = self.heatmaps_loss(
            stages_pred_heatmaps, stages_target_heatmaps, target_weights
        )
        ae_grouping_loss = self.tags_loss(stages_pred_tags, target_keypoints)
        return heamtaps_loss + ae_grouping_loss
