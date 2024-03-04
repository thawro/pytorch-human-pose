import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


class HeatmapsLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_heatmaps: Tensor, target_heatmaps: Tensor, mask: Tensor) -> Tensor:
        loss = ((pred_heatmaps - target_heatmaps) ** 2) * mask[:, None, :, :].expand_as(
            pred_heatmaps
        )
        return loss.mean()


class AEGroupingLoss(_Loss):
    def forward(self, pred_tags: Tensor, joints: list[Tensor]):
        batch_size = len(joints)
        pull_loss = 0
        push_loss = 0
        for i in range(batch_size):
            tags = []  # ^h_n
            pull = 0  # intra object kpts
            for j, obj_joints in enumerate(joints[i]):
                obj_tags = []
                for k, joint in enumerate(obj_joints):
                    x, y, vis = joint
                    if vis > 0:
                        obj_tags.append(pred_tags[i, k, y, x])
                if len(obj_tags) == 0:
                    continue
                obj_tags = torch.stack(obj_tags)
                obj_ref_tag = obj_tags.mean()
                tags.append(obj_ref_tag)

                pull = pull + ((obj_tags - obj_ref_tag.expand_as(obj_tags)) ** 2).mean()

            num_obj = len(tags)

            if num_obj == 0:
                continue
            elif num_obj == 1:
                pull_loss = pull_loss + pull / num_obj
                continue
            pull_loss = pull_loss + pull / num_obj
            tags = torch.stack(tags)

            size = (num_obj, num_obj)
            A = tags.expand(*size)
            B = A.permute(1, 0)

            diff = A - B

            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_obj
            push = push / ((num_obj - 1) * num_obj) * 0.5
            push_loss = push_loss + push
        return push_loss / batch_size, pull_loss / batch_size


class AEKeypointsLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()
        self.heatmaps_losses = nn.ModuleList([HeatmapsLoss() for _ in range(2)])
        self.tags_loss = AEGroupingLoss()

    def calculate_loss(
        self,
        stages_pred_kpts_heatmaps: list[Tensor],
        pred_tags_heatmaps: Tensor,
        stages_target_heatmaps: list[Tensor],
        masks: list[Tensor],
        joints: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        num_stages = len(stages_target_heatmaps)
        heatmap_losses = []
        push_losses = []
        pull_losses = []

        # heatmaps_loss = 0
        for i in range(num_stages):
            hm_loss = self.heatmaps_losses[i](
                stages_pred_kpts_heatmaps[i], stages_target_heatmaps[i], masks[i]
            )
            heatmap_losses.append(hm_loss)
            # heatmaps_loss = heatmaps_loss + hm_loss
        push_loss, pull_loss = self.tags_loss(pred_tags_heatmaps, joints[0])
        push_losses.append(push_loss * 1e-3)
        pull_losses.append(pull_loss * 1e-3)
        return heatmap_losses, push_losses, pull_losses
