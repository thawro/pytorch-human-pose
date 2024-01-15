import numpy as np
import torch
from munkres import Munkres
from collections import defaultdict
from torch import Tensor


def py_max_match(scores: np.ndarray) -> np.ndarray:
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


JOINTS_ORDER = [
    i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
]


class SPPEHeatmapParser:
    def __init__(self, num_kpts: int, det_thr: float = 0.2):
        self.num_kpts = num_kpts
        self.det_thr = det_thr

    def match(self, heatmaps: Tensor) -> np.ndarray:
        """
        heatmaps: detection heatmaps. Tensor of shape [num_kpts, height, width]

        Return joints array of shape [num_person, num_kpts, 3], where 3 is for
        (x, y, score) of each keypoint
        """
        num_person = 1
        num_kpts, h, w = heatmaps.shape
        joints = torch.zeros(num_person, num_kpts, 3)
        flat_heatmaps = heatmaps.view(num_kpts, -1)
        coords = torch.argmax(flat_heatmaps, 1)
        coords = coords.view(num_kpts, 1) + 1
        coords = coords.repeat(1, 2).float()
        coords[:, 0] = (coords[:, 0] - 1) % w
        coords[:, 1] = torch.floor((coords[:, 1] - 1) / w)
        coords = coords.to(torch.int32)  # .flip(-1)
        # coords are in [x, y] order
        scores = torch.zeros(num_kpts)
        for idx in range(num_kpts):
            x, y = coords[idx].tolist()
            scores[idx] = heatmaps[idx][y, x]
        joints[..., :2] = coords
        joints[..., 2] = scores
        return joints.cpu().numpy()

    def parse(self, heatmaps: Tensor) -> np.ndarray:
        """
        heatmaps: detection heatmaps. Tensor of shape [1, num_kpts, height, width]

        Return joints array of shape [1, num_kpts, 3], where 1 is for person number and 3 is for
        (x, y, score) of each keypoint
        """
        heatmaps = heatmaps[0]
        joints = self.match(heatmaps)
        mask = joints[..., 2] < self.det_thr
        joints[mask][..., :2] = (0, 0)
        return joints


class MPPEHeatmapParser:
    def __init__(
        self,
        num_kpts: int,
        max_num_people: int = 3,
        det_thr: float = 0.2,
        tag_thr: float = 1.0,
    ):
        self.pool = torch.nn.MaxPool2d(3, 1, 1)
        self.max_num_people = max_num_people
        self.num_kpts = num_kpts
        self.joints_order = JOINTS_ORDER
        self.det_thr = det_thr
        self.tag_thr = tag_thr

    def nms(self, heatmaps: Tensor) -> Tensor:
        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm
        return heatmaps

    def match(self, joints_tags, joints_coords, joints_scores):
        """
        joints_tags are tags for each joint for each person detected, shape: [num_kpts, num_person, embedding_dim]
        joints_coords are yx coordinates for each joint for each person detected, shape: [num_kpts, num_person, 2]
        joints_scores are scores (from heatmaps) for each joint for each person detected, shape: [num_kpts, num_person]
        """
        joints_tags = joints_tags.round()  # TODO
        joints_scores = np.expand_dims(joints_scores, -1)

        joints_dim = sum(
            arr.shape[-1] for arr in [joints_tags, joints_scores, joints_coords]
        )
        joint_dict = defaultdict(lambda: np.zeros((self.num_kpts, joints_dim)), {})
        tag_dict = {}
        for i in range(self.num_kpts):
            idx = self.joints_order[i]

            joint_tags = joints_tags[idx]  # shape: [num_person, embedding_dim]
            joint_coords = joints_coords[idx]  # shape: [num_person, 2]
            joint_scores = joints_scores[idx]  # shape: [num_person, 1]

            # shape: [num_person, joints_dim]
            joints = np.concatenate((joint_coords, joint_scores, joint_tags), 1)

            mask = joint_scores.squeeze() > self.det_thr
            joints = joints[mask]
            joint_tags = joint_tags[mask]

            if mask.sum() == 0:
                continue

            if len(joint_dict) == 0:
                for tag_idx, tag in enumerate(joint_tags):
                    joint = joints[tag_idx]
                    # getting 0th element in case of multidim tag embedding
                    key = tag[0]  # TODO
                    joint_dict[key][idx] = joint.copy()
                    tag_dict[key] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[: self.max_num_people]
                grouped_tags = np.array(
                    [np.mean(tag_dict[key], axis=0) for key in grouped_keys]
                )
                diff = np.expand_dims(joint_tags, 1) - np.expand_dims(grouped_tags, 0)
                diff = np.linalg.norm(diff, ord=2, axis=2)

                _diff = np.copy(diff)

                # diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

                n_added = len(joint_tags)
                n_grouped = len(grouped_keys)
                if n_added > n_grouped:  # some new keypoints appeared as detected
                    diff = np.concatenate(
                        (diff, np.zeros((n_added, n_added - n_grouped)) + 1e10), axis=1
                    )

                pairs = py_max_match(diff)

                for row, col in pairs:
                    if (
                        row < n_added
                        and col < n_grouped
                        and _diff[row][col] < self.tag_thr
                    ):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(joint_tags[row])
                    else:
                        key = joint_tags[row][0]
                        joint_dict[key][idx] = joints[row].copy()
                        tag_dict[key] = [joint_tags[row]]

        joints = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
        return joints  # [: self.max_num_people]

    def top_k(
        self, heatmaps: Tensor, tags: Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, num_kpts, h, w = heatmaps.shape

        heatmaps = self.nms(heatmaps)

        # tag_emb_dim = tags.shape[-1]

        heatmaps = heatmaps.view(batch_size, num_kpts, -1)
        tags = tags.view(batch_size, num_kpts, w * h, -1)
        tag_emb_dim = tags.shape[-1]

        scores_k, idxs_k = heatmaps.topk(self.max_num_people, dim=2)

        x = idxs_k % w
        y = (idxs_k / w).long()
        coords_k = torch.stack((x, y), dim=3)

        tags_k = torch.stack(
            [torch.gather(tags[:, :, :, i], 2, idxs_k) for i in range(tag_emb_dim)],
            dim=3,
        )

        joints_tags = tags_k.detach().cpu().numpy()
        joints_coords = coords_k.detach().cpu().numpy()
        joints_scores = scores_k.detach().cpu().numpy()
        # joints_tags = tags_k.numpy()
        # joints_coords = coords_k.numpy()
        # joints_scores = scores_k.numpy()
        return joints_tags, joints_coords, joints_scores

    def adjust(self, joints: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
        num_person, num_kpts, _ = joints.shape
        num_kpts, h, w = heatmaps.shape
        for person_idx, person_joints in enumerate(joints):
            for joint_idx, joint in enumerate(person_joints):
                y, x, score = joint[:3]
                if score > 0:
                    xx, yy = int(x), int(y)
                    tmp = heatmaps[joint_idx]
                    if tmp[xx, min(yy + 1, w - 1)] > tmp[xx, max(yy - 1, 0)]:
                        y += 0.25
                    else:
                        y -= 0.25

                    if tmp[min(xx + 1, h - 1), yy] > tmp[max(0, xx - 1), yy]:
                        x += 0.25
                    else:
                        x -= 0.25
                    new_coords = (y + 0.5, x + 0.5)
                    joints[person_idx, joint_idx, 0:2] = new_coords
        return joints

    def refine(
        self, heatmaps: np.ndarray, tags: np.ndarray, joints: np.ndarray
    ) -> np.ndarray:
        """
        For specific person preds
        Given initial keypoint predictions, identify missing joints
        heatmaps: detection heatmaps, shape: [num_kpts, h, w]
        tags: tags heatmaps, shape: [num_kpts, h, w]
        joints: joints array, shape [num_kpts, 4]  , 4 is for (x, y, score, tag)
        """
        num_kpts, h, w = heatmaps.shape
        if len(tags.shape) == 3:
            tags = np.expand_dims(tags, -1)

        _tags = []
        for i in range(num_kpts):
            if joints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = joints[i, :2].astype(np.int32)
                _tags.append(tags[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(_tags, axis=0)
        prev_tag = np.expand_dims(prev_tag, (0, 1))

        _joints = []
        for i in range(num_kpts):
            # score of joints i at all position
            joint_hm = heatmaps[i]
            # distance of all tag values with mean tag of current detected people
            tags_dist = ((tags[i] - prev_tag) ** 2).sum(axis=2) ** 0.5
            hm_tags_diff = joint_hm - np.round(tags_dist)

            # find maximum position
            y, x = np.unravel_index(np.argmax(hm_tags_diff), joint_hm.shape)
            xx = x
            yy = y
            # detection score at maximum position
            score = joint_hm[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if joint_hm[yy, min(xx + 1, w - 1)] > joint_hm[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if joint_hm[min(yy + 1, h - 1), xx] > joint_hm[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            _joints.append((x, y, score))

        _joints = np.array(_joints)
        if len(_joints) > 0:
            for i in range(num_kpts):
                # add keypoint if it is not detected
                if _joints[i, 2] > 0 and joints[i, 2] == 0:
                    joints[i, :3] = _joints[i]
        return joints

    def parse(
        self, heatmaps: Tensor, tags: Tensor, adjust: bool = True, refine: bool = True
    ) -> np.ndarray:
        """
        heatmaps: detection heatmaps. Tensor of shape [1, num_kpts, height, width]
        tags: tags heatmaps. Tensor of shape [1, num_kpts, height, width]
        adjust: whether to adjust for quantization
        refine: whether to refine missing joints

        Return joints array of shape [num_person, num_kpts, 4], where 4 is for
        (x, y, score, tag) of each keypoint
        """
        num_kpts = heatmaps.shape[1]
        heatmaps_npy = heatmaps.detach().cpu().numpy()[0]
        tags_npy = tags.detach().cpu().numpy()[0]

        joints_tags, joints_coords, joints_scores = self.top_k(heatmaps, tags)
        joints_tags = joints_tags[0]
        joints_coords = joints_coords[0]
        joints_scores = joints_scores[0]

        joints = self.match(joints_tags, joints_coords, joints_scores)

        if len(joints) == 0:
            joints = np.zeros((1, num_kpts, 4))
            return joints

        if adjust:
            joints = self.adjust(joints, heatmaps_npy)

        num_person = len(joints)
        if refine:
            for i in range(num_person):
                joints[i] = self.refine(heatmaps_npy, tags_npy, joints[i])

        return joints


if __name__ == "__main__":
    from src.utils.config import DS_ROOT
    from src.keypoints.transforms import MPPEKeypointsTransform
    from src.keypoints.datasets import MPPEKeypointsDataset
    from geda.data_providers.mpii import LABELS, LIMBS
    import torch
    import cv2

    split = "train"

    parser = MPPEHeatmapParser(max_num_people=3, num_kpts=16)

    transform = MPPEKeypointsTransform(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], out_size=(512, 512)
    )

    ds_root = str(DS_ROOT / "MPII" / "HumanPose")

    hm_resolutions = [1 / 2, 1 / 4]

    ds = MPPEKeypointsDataset(
        ds_root, split, transform, hm_resolutions, labels=LABELS, limbs=LIMBS
    )

    def run_grouping(idx):
        image, scales_heatmaps, target_weights, keypoints, visibilities = ds[idx]

        heatmaps = torch.from_numpy(scales_heatmaps[0]).unsqueeze(0)
        tags = torch.ones_like(heatmaps) * 3
        joints = parser.parse(heatmaps, tags, True, False)

    ds.explore(idx=13, callback=run_grouping)
