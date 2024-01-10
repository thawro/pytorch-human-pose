from torch import nn, Tensor
from src.base.loss import BaseLoss, WeightedLoss
from torch.nn.modules.loss import _Loss
import torch


class HeatmapsLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        target_weights: Tensor,
    ) -> Tensor:
        # TODO: sth not working right , maybe data parallel?
        # target_masks = target_weights > 0
        # return self.criterion(
        #     pred_heatmaps[target_masks], target_heatmaps[target_masks]
        # )
        return (
            self.criterion(pred_heatmaps, target_heatmaps).mean((2, 3)) * target_weights
        ).mean()


class StagesHeatmapsLoss(_Loss):
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
        super().__init__(WeightedLoss(StagesHeatmapsLoss(), weight=1))

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
    def __init__(self, hm_resolution: float):
        super().__init__()
        self.hm_resolution = hm_resolution

    def forward(
        self, pred_tags: Tensor, target_keypoints: list, target_visibilities: Tensor
    ):
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
                    _x = int(x * self.hm_resolution)
                    _y = int(y * self.hm_resolution)
                    if target_visibilities[i][j][k] > 0:  # is visible
                        tag = tags[k][_y, _x]
                        obj_kpts_tags.append(tag)
                if len(obj_kpts_tags) == 0:
                    continue
                # TODO: check if gradient flow is correct\
                obj_kpts_tags = torch.stack(obj_kpts_tags)
                obj_ref_tag = obj_kpts_tags.mean()
                all_objs_ref_tags.append(obj_ref_tag)

                obj_pull_loss = ((obj_kpts_tags - obj_ref_tag) ** 2).mean()
                all_objs_pull_loss += obj_pull_loss

            num_obj = len(all_objs_ref_tags)

            if num_obj == 0:
                continue
            elif num_obj == 1:
                pull_loss += all_objs_pull_loss / num_obj
            else:
                pull_loss += all_objs_pull_loss / num_obj
                ref_tags = torch.stack(all_objs_ref_tags)

                size = (num_obj, num_obj)
                A = ref_tags.expand(*size)
                B = A.permute(1, 0)

                diff = A - B

                diff = torch.pow(diff, 2)
                _push_loss = torch.exp(-diff)
                _push_loss = torch.sum(_push_loss)  # - num_obj
                _push_loss = _push_loss / ((num_obj - 1) * num_obj) * 0.5
                push_loss += _push_loss
        return (push_loss + pull_loss) / batch_size


class AEKeypointsLoss(_Loss):
    def __init__(self, hm_resolutions: list[float]) -> None:
        super().__init__()
        self.heatmaps_loss = WeightedLoss(HeatmapsLoss(), weight=1)
        self.tags_losses = [
            WeightedLoss(AEGroupingLoss(hm_res), weight=1e-3)
            for hm_res in hm_resolutions
        ]

    def calculate_loss(
        self,
        stages_pred_kpts_heatmaps: list[Tensor],
        stages_pred_tags_heatmaps: list[Tensor],
        stages_target_heatmaps: list[Tensor],
        target_weights: Tensor,
        target_keypoints: list,
        target_visibilities: list,
    ) -> tuple[Tensor, Tensor]:
        num_stages = len(stages_target_heatmaps)
        heatmaps_loss = 0
        ae_grouping_loss = 0
        for i in range(num_stages):
            hm_loss = self.heatmaps_loss(
                stages_pred_kpts_heatmaps[i], stages_target_heatmaps[i], target_weights
            )

            ae_loss = self.tags_losses[i](
                stages_pred_tags_heatmaps[i], target_keypoints, target_visibilities
            )
            heatmaps_loss += hm_loss
            ae_grouping_loss += ae_loss
        return heatmaps_loss, ae_grouping_loss


def test():
    from src.keypoints.datasets import MppeMpiiDataset
    from src.keypoints.transforms import MPPEKeypointsTransform
    import cv2
    from src.utils.config import DS_ROOT
    import random

    def explore(idx, ds):
        (
            image,
            scales_heatmaps,
            target_weights,
            keypoints,
            visibilities,
            extra_coords,
        ) = ds[idx]

        # TODO: change tag values and see how loss behaves

        scales_heatmaps = [torch.from_numpy(scales_heatmaps[0]).unsqueeze(0)]
        target_weights = torch.from_numpy(target_weights).unsqueeze(0)
        keypoints = [keypoints]
        visibilities = [visibilities]
        extra_coords = [extra_coords]
        pred_tags = torch.zeros(1, 16, res // 4, res // 4)

        num_obj = len(keypoints[0])

        tag = 0
        for i in range(num_obj):
            kpts = keypoints[0][i]
            for k, (x, y) in enumerate(kpts):
                pred_tags[0][k][y // 4, x // 4] = tag + random.random() * 10
            tag += 1

        image = cv2.cvtColor(transform.inverse_preprocessing(image), cv2.COLOR_RGB2BGR)

        for j, obj_kpts in enumerate(keypoints[0]):
            obj_kpts_tags = []
            for k, kpt in enumerate(obj_kpts):
                x, y = kpt
                # coords were wrt image size, now we need to parse it to tags size
                # x, y = x // 4, y // 4
                if x > 0 and y > 0 and x < res and y < res:  # is visible
                    tag = pred_tags[0][k][y // 4, x // 4]
                    obj_kpts_tags.append(tag)
                    cv2.circle(image, (x, y), 3, (50, 155, 50), -1)

        cv2.imshow("Image", image)

        loss_fn = AEKeypointsLoss()
        loss = loss_fn.calculate_loss(
            scales_heatmaps,
            scales_heatmaps,
            pred_tags,
            target_weights,
            keypoints,
        )
        k = cv2.waitKeyEx(0)
        # change according to your system
        left_key = 65361
        right_key = 65363
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing")
            cv2.destroyAllWindows()
            return
        elif k % 256 == 32 or k == right_key:  # SPACE or right arrow pressed
            print("Space or right arrow hit, exploring next sample")
            idx += 1
        elif k == left_key:  # SPACE or right arrow pressed
            print("Left arrow hit, exploring previous sample")
            idx -= 1
        explore(idx, ds)

    hm_resolutions = [1 / 2]
    tags_resolution = 1 / 4
    res = 512

    ds_root = str(DS_ROOT / "MPII" / "HumanPose")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = MPPEKeypointsTransform(mean=mean, std=std, out_size=(res, res))

    ds = MppeMpiiDataset(ds_root, "val", transform, hm_resolutions)

    explore(22, ds)


if __name__ == "__main__":
    test()
