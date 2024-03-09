"""
Based on https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/core/group.py
"""

import numpy as np
import torch
from munkres import Munkres


class SPPEHeatmapParser:
    def __init__(self, num_kpts: int, det_thr: float = 0.2):
        self.num_kpts = num_kpts
        self.det_thr = det_thr

    def match(self, heatmaps: torch.Tensor) -> np.ndarray:
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

    def parse(self, heatmaps: torch.Tensor) -> np.ndarray:
        """
        heatmaps: detection heatmaps. Tensor of shape [1, num_kpts, height, width]

        Return joints array of shape [1, num_kpts, 3], where 1 is for person number and 3 is for
        (x, y, score) of each keypoint
        """
        heatmaps = heatmaps[0]
        joints = self.match(heatmaps)
        # mask = joints[..., 2] < self.det_thr
        # joints[mask][..., :2] = (0, 0)
        return joints


def py_max_match(scores: np.ndarray):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


class MPPEHeatmapParser(object):
    joints_order: list[int] = [
        i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
    ]

    def __init__(
        self,
        num_kpts: int,
        max_num_people: int = 30,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
    ):
        self.pool = torch.nn.MaxPool2d(5, 1, 2)
        self.max_num_people = max_num_people
        self.num_kpts = num_kpts
        self.det_thr = det_thr
        self.tag_thr = tag_thr

    def nms(self, kpts_heatmaps: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(kpts_heatmaps)
        pooled = torch.eq(pooled, kpts_heatmaps).float()
        return kpts_heatmaps * pooled

    def match_by_tag(
        self, tags_k: np.ndarray, coords_k: np.ndarray, scores_k: np.ndarray
    ) -> np.ndarray:
        """
        Grouping by tag from: https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
        """
        # 3 from: coords (2), score (1)
        default_ = np.zeros((self.num_kpts, 3 + tags_k.shape[2]))

        joint_dict = {}
        tag_dict = {}
        for i in range(self.num_kpts):
            idx = self.joints_order[i]
            tags = tags_k[idx]
            joints = np.concatenate((coords_k[idx], scores_k[idx, :, None], tags), 1)
            mask = joints[:, 2] > self.det_thr
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(joint_dict) == 0:
                for tag, joint in zip(tags, joints):
                    key = tag[0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                    tag_dict[key] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[: self.max_num_people]
                grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

                diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
                diff_normed = np.linalg.norm(diff, ord=2, axis=2)
                diff_saved = np.copy(diff_normed)

                use_detection_val = True
                if use_detection_val:
                    diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

                num_added, num_grouped = diff.shape[:2]

                if num_added > num_grouped:
                    huge_diff = np.zeros((num_added, num_added - num_grouped)) + 1e10
                    diff_normed = np.concatenate((diff_normed, huge_diff), axis=1)

                pairs = py_max_match(diff_normed)
                for row, col in pairs:
                    if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < self.tag_thr
                    ):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(tags[row])
                    else:
                        key = tags[row][0]
                        joint_dict.setdefault(key, np.copy(default_))[idx] = joints[row]
                        tag_dict[key] = [tags[row]]
        grouped_joints = np.array(list(joint_dict.values())).astype(np.float32)
        return grouped_joints[: self.max_num_people]  # TODO: check if good

    def top_k(
        self, kpts_hms: torch.Tensor, tags_hms: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kpts_hms = self.nms(kpts_hms.unsqueeze(0))[0]
        num_kpts, h, w = kpts_hms.shape
        kpts_hms = kpts_hms.view(num_kpts, -1)
        scores_k, idxs = kpts_hms.topk(self.max_num_people, dim=1)

        tags_hms = tags_hms.view(num_kpts, w * h, -1)
        tags_emb_dim = tags_hms.size(2)

        tags_k = torch.stack(
            [torch.gather(tags_hms[..., i], 1, idxs) for i in range(tags_emb_dim)],
            dim=2,
        )

        x = idxs % w
        y = (idxs / w).long()
        coords_k = torch.stack((x, y), dim=2)

        tags_k = tags_k.cpu().numpy()
        coords_k = coords_k.cpu().numpy().astype(np.int32)
        scores_k = scores_k.cpu().numpy()
        return tags_k, coords_k, scores_k

    def adjust(self, grouped_joints: np.ndarray, kpts_hms: np.ndarray) -> np.ndarray:
        # quarter offset adjustment
        h, w = kpts_hms.shape[-2:]
        for person_idx, person_joints in enumerate(grouped_joints):
            for joint_idx, (y, x, score, *tag) in enumerate(person_joints):
                if score == 0:
                    continue
                xx, yy = int(x), int(y)
                kpt_hm = kpts_hms[joint_idx]
                if kpt_hm[xx, min(yy + 1, w - 1)] > kpt_hm[xx, max(yy - 1, 0)]:
                    y += 0.25
                else:
                    y -= 0.25

                if kpt_hm[min(xx + 1, h - 1), yy] > kpt_hm[max(0, xx - 1), yy]:
                    x += 0.25
                else:
                    x -= 0.25
                grouped_joints[person_idx, joint_idx, :2] = (y + 0.5, x + 0.5)
        return grouped_joints

    def refine(
        self, kpts_hms: np.ndarray, tags_hms: np.ndarray, person_joints: np.ndarray
    ) -> np.ndarray:
        """
        Given initial keypoint predictions, we identify missing joints
        :param kpts_hms: numpy.ndarray of size (num_kpts, H, W)
        :param tags_hms: numpy.ndarray of size (num_kpts, H, W)
        :param person_joints: numpy.ndarray of size (num_kpts, 4) ,last dim is (x, y, score, tag_emb)
        """
        h, w = kpts_hms.shape[-2:]
        if len(tags_hms.shape) == 3:
            tags_hms = tags_hms[..., None]
        tags = []
        for i in range(self.num_kpts):
            if person_joints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = person_joints[i][:2].astype(np.int32)
                tags.append(tags_hms[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)[None, None, :]
        tmp_joints = []

        for i in range(self.num_kpts):
            # score of joints i at all position
            kpt_hm = kpts_hms[i]

            # distance of all tag values with mean tag of current detected people
            tags_dist = ((tags_hms[i] - prev_tag) ** 2).sum(axis=2) ** 0.5
            hms_diff = kpt_hm - np.round(tags_dist)

            # find maximum position
            y, x = np.unravel_index(np.argmax(hms_diff), (h, w))
            xx, yy = x, y

            # detection score at maximum position
            val = kpt_hm[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if kpt_hm[yy, min(xx + 1, w - 1)] > kpt_hm[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if kpt_hm[min(yy + 1, h - 1), xx] > kpt_hm[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            tmp_joints.append((x, y, val))
        tmp_joints = np.array(tmp_joints)

        replace_mask = np.bitwise_and(tmp_joints[:, 2] > 0, person_joints[:, 2] == 0)
        person_joints[replace_mask, :3] = tmp_joints[replace_mask]
        return person_joints

    def parse(
        self,
        kpts_hms: torch.Tensor,
        tags_hms: torch.Tensor,
        adjust: bool = True,
        refine: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # kpts_hms, tags_hms = kpts_hms[0], tags_hms[0]
        tags_k, coords_k, scores_k = self.top_k(kpts_hms, tags_hms)
        grouped_joints = self.match_by_tag(tags_k, coords_k, scores_k)
        if len(grouped_joints) == 0:  # take only best pred
            coords = coords_k[:, 0]
            score = np.expand_dims(scores_k[:, 0], -1)
            tag = tags_k[:, 0]
            grouped_joints = np.concatenate([coords, score, tag], axis=-1)
            grouped_joints = np.expand_dims(grouped_joints, 0)
            grouped_joints = np.nan_to_num(grouped_joints, nan=0)
            grouped_joints[..., 2] = 0.01

        kpts_hms_npy = kpts_hms.cpu().numpy()
        tags_hms_npy = tags_hms.cpu().numpy()

        if adjust:
            grouped_joints = self.adjust(grouped_joints, kpts_hms_npy)
        person_scores = grouped_joints[..., 2].mean(1)

        if refine:
            for person_idx in range(len(grouped_joints)):
                grouped_joints[person_idx] = self.refine(
                    kpts_hms_npy, tags_hms_npy, grouped_joints[person_idx]
                )
        return grouped_joints, person_scores
