import glob
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pycocotools
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.keypoints.transforms import ComposeKeypointsTransform, KeypointsTransform
from src.keypoints.utils import coco_polygons_to_mask, coco_rle_to_seg, mask_to_polygons
from src.keypoints.visualization import plot_connections, plot_heatmaps
from src.logger.pylogger import log
from src.utils.files import load_yaml, save_yaml
from src.utils.image import get_color, make_grid, put_txt, stack_horizontally
from src.utils.utils import get_rank

COCO_LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_LIMBS = [
    (9, 7),
    (7, 5),
    (5, 3),
    (3, 1),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def get_coco_joints(annots: list[dict]):
    num_people = len(annots)
    num_kpts = 17
    joints = np.zeros((num_people, num_kpts, 3))
    for i, obj in enumerate(annots):
        joints[i] = np.array(obj["keypoints"]).reshape([-1, 3])
    return joints


class HeatmapGenerator:
    """
    source: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
    """

    def __init__(self, num_kpts: int, size: int, sigma: float = 2):
        self.num_kpts = num_kpts
        self.size = size
        self.h, self.w = size, size
        if sigma < 0:
            sigma = size / 64
        self.sigma = sigma
        x = np.arange(0, 6 * sigma + 3, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(
        self,
        joints: np.ndarray,
        # max_h: int,
        # max_w: int,
    ) -> np.ndarray:
        """
        visibility: 0 - not labeled, 1 - labeled but not visible, 2 - labeled and visible
        """
        hms = np.zeros((self.num_kpts, self.h, self.w), dtype=np.float32)
        for joint in joints:
            for idx in range(self.num_kpts):
                x, y, vis = joint[idx]
                if vis <= 0 or x < 0 or y < 0 or x >= self.w or y >= self.h:
                    continue

                xmin = int(np.round(x - 3 * self.sigma - 1))
                ymin = int(np.round(y - 3 * self.sigma - 1))
                xmax = int(np.round(x + 3 * self.sigma + 2))
                ymax = int(np.round(y + 3 * self.sigma + 2))

                c, d = max(0, -xmin), min(xmax, self.w) - xmin
                a, b = max(0, -ymin), min(ymax, self.h) - ymin

                cc, dd = max(0, xmin), min(xmax, self.w)
                aa, bb = max(0, ymin), min(ymax, self.h)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.gauss[a:b, c:d])
        return hms


class JointsGenerator:
    def __init__(self, size: int = 512):
        self.h, self.w = size, size

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        for i in range(len(joints)):
            for k, pt in enumerate(joints[i]):
                x, y, vis = int(pt[0]), int(pt[1]), pt[2]
                if vis > 0 and x >= 0 and y >= 0 and x < self.w and y < self.h:
                    joints[i, k] = (x, y, 1)
                else:
                    joints[i, k] = (0, 0, 0)
        visible_joints_mask = joints.sum(axis=(1, 2)) > 0
        return joints[visible_joints_mask].astype(np.int32)


def collate_fn(
    batch: list[tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]],
) -> tuple[
    Tensor,
    list[Tensor],
    list[Tensor],
    list[list[np.ndarray]],
]:
    num_scales = len(batch[0][1])
    images = torch.from_numpy(np.stack([item[0] for item in batch]))
    heatmaps_tensor = []
    masks_tensor = []
    joints_scales = []
    for i in range(num_scales):
        _heatmaps = []
        _masks = []
        _joints = []
        for sample in batch:
            _heatmaps.append(sample[1][i])
            _masks.append(sample[2][i])
            _joints.append(sample[3][i])
        heatmaps_tensor.append(torch.from_numpy(np.stack(_heatmaps)))
        masks_tensor.append(torch.from_numpy(np.stack(_masks)))
        joints_scales.append(_joints)
    return images, heatmaps_tensor, masks_tensor, joints_scales


def get_crowd_mask(annot: list, img_h: int, img_w: int) -> np.ndarray:
    m = np.zeros((img_h, img_w))
    for obj in annot:
        if obj["iscrowd"]:
            rle = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            m += pycocotools.mask.decode(rle)
        elif obj["num_keypoints"] == 0:
            rles = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            for rle in rles:
                m += pycocotools.mask.decode(rle)
    return m < 0.5


class CocoKeypointsDataset(BaseImageDataset):
    limbs = COCO_LIMBS
    labels = COCO_LABELS
    name: str = "COCO"

    def __init__(
        self,
        root: str,
        split: str,
        transform: ComposeKeypointsTransform | None = None,
        out_size: int = 512,
        hm_resolutions: list[float] = [1 / 4, 1 / 2],
        num_kpts: int = 17,
        max_num_people: int = 30,
        sigma: float = 2,
        mosaic_probability: float = 0,
    ):
        self.root = root
        self.split = split
        kpts_dir = f"person_keypoints_{self.split}"
        self.images_dir = f"{self.root}/images/{self.split}"
        self.annots_dir = f"{self.root}/annotations/{kpts_dir}"
        self.masks_dir = f"{self.root}/masks/{kpts_dir}"
        self.out_size = out_size
        self.is_train = "train" in split
        self.num_scales = len(hm_resolutions)
        self.num_kpts = num_kpts
        self.max_num_people = max_num_people
        self.mosaic_probability = mosaic_probability
        self._save_annots_to_files()
        self._set_paths()
        self.transform = transform
        self.hm_resolutions = hm_resolutions
        self.hm_sizes = [int(res * out_size) for res in hm_resolutions]
        hm_generators = []
        joints_generator = []
        for hm_size in self.hm_sizes:
            hm_generators.append(HeatmapGenerator(self.num_kpts, hm_size, sigma=sigma))
            joints_generator.append(JointsGenerator(hm_size))
        self.hm_generators = hm_generators
        self.joints_generators = joints_generator

    def _set_paths(self):
        kpts_dir = f"person_keypoints_{self.split}"
        annots_filepaths = glob.glob(f"{self.annots_dir}/*")
        images_filepaths = [
            path.replace(".yaml", ".jpg").replace(f"annotations/{kpts_dir}", f"images/{self.split}")
            for path in annots_filepaths
        ]
        masks_filepaths = [
            path.replace(".yaml", ".npy").replace("annotations/", "masks/")
            for path in annots_filepaths
        ]
        self.annots_filepaths = np.array(annots_filepaths, dtype=np.str_)
        self.images_filepaths = np.array(images_filepaths, dtype=np.str_)
        self.masks_filepaths = np.array(masks_filepaths, dtype=np.str_)
        num_imgs = len(images_filepaths)
        num_annots = len(annots_filepaths)
        num_masks = len(masks_filepaths)
        assert (
            num_imgs == num_annots == num_masks
        ), f"There must be the same number of images and annotations. Currently: num_imgs={num_imgs}, num_annots={num_annots}, num_masks={num_masks}"

    def _save_annots_to_files(self):
        # save mask to npy file and annot to yaml file
        rank = get_rank()
        if rank != 0:
            log.warn(
                f"     Current process (rank = {rank}) is not the main process (rank = 0) -> Skipping annots files saving"
            )
            return
        if os.path.exists(self.annots_dir):
            log.info(
                f"..{self.split} annotations already saved to files -> Skipping annots files saving.."
            )
            return
        log.info(f"..Saving {self.split} annotations (keypoints and crowd masks) to files..")
        Path(self.annots_dir).mkdir(exist_ok=True, parents=True)
        Path(self.masks_dir).mkdir(exist_ok=True, parents=True)
        coco = COCO(f"{self.root}/annotations/person_keypoints_{self.split}.json")
        ids = list(coco.imgs.keys())
        # remove_images_without_annotations
        ids = [_id for _id in ids if len(coco.getAnnIds(imgIds=_id, iscrowd=None)) > 0]
        for idx in tqdm(range(len(ids)), desc=f"Saving {self.split} annotations"):
            img_id = ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annot = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            img_filepath = f"{self.root}/images/{self.split}/{img_info['file_name']}"
            filename = Path(img_filepath).stem
            annot_filepath = f"{self.annots_dir}/{filename}.yaml"
            mask_filepath = f"{self.masks_dir}/{filename}.npy"
            mask = get_crowd_mask(annot, img_info["height"], img_info["width"])
            np.save(mask_filepath, mask)
            save_yaml(annot, annot_filepath)

    def load_annot(self, idx: int) -> dict | list[dict]:
        return load_yaml(self.annots_filepaths[idx])

    def get_raw_data(self, idx: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        mask = np.load(self.masks_filepaths[idx])
        return image, annot, mask

    def get_raw_mosaiced_data(
        self, idx: int, add_segmentation: bool = False
    ) -> tuple[np.ndarray, list[dict], np.ndarray]:
        out_size = self.out_size * 2
        img_size = out_size // 2
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        mosaic_annot = []
        mosaic_img = np.zeros([out_size, out_size, 3], dtype=np.uint8)
        mosaic_mask = np.empty([out_size, out_size], dtype=np.bool_)

        new_h, new_w = img_size, img_size

        for i in range(4):
            idx = idxs[i]
            img, annot, mask = self.get_raw_data(idx)
            img_h, img_w = img.shape[:2]

            if i == 0:  # top-left
                s_y, s_x = 0, 0
            elif i == 1:  # top-right
                s_y, s_x = 0, new_w
            elif i == 2:  # bottom-left
                s_y, s_x = new_h, 0
            else:
                s_y, s_x = new_h, new_w

            new_img = cv2.resize(img, (new_w, new_h))
            new_mask = cv2.resize((mask * 255).astype(np.uint8), (new_w, new_h)) > 0.5

            scale_y, scale_x = new_h / img_h, new_w / img_w
            objects = []
            for obj in annot:
                bbox = obj["bbox"]
                bbox[0] = int(bbox[0] * scale_x + s_x)
                bbox[2] = int(bbox[2] * scale_x + s_x)
                bbox[1] = int(bbox[1] * scale_y + s_y)
                bbox[3] = int(bbox[3] * scale_y + s_y)
                kpts = np.array(obj["keypoints"]).reshape([-1, 3])
                vis_mask = kpts[:, 2] <= 0
                kpts[:, 0] = kpts[:, 0] * scale_x + s_x
                kpts[:, 1] = kpts[:, 1] * scale_y + s_y
                kpts[vis_mask] = kpts[vis_mask] * 0

                segmentation = None
                if add_segmentation:
                    segmentation = obj["segmentation"]
                    if isinstance(segmentation, dict):
                        segmentation = mask_to_polygons(
                            coco_polygons_to_mask(segmentation, img_h, img_w)
                        )
                    for j in range(len(segmentation)):
                        seg = np.array(segmentation[j])
                        seg[::2] = seg[::2] * scale_x + s_x
                        seg[1::2] = seg[1::2] * scale_y + s_y
                        segmentation[j] = seg.astype(np.int32).tolist()

                objects.append(
                    {
                        "bbox": bbox,
                        "iscrowd": obj["iscrowd"],
                        "keypoints": kpts,
                        "num_keypoints": obj["num_keypoints"],
                        "segmentation": segmentation,
                    }
                )

            mosaic_img[s_y : s_y + new_h, s_x : s_x + new_w] = new_img
            mosaic_mask[s_y : s_y + new_h, s_x : s_x + new_w] = new_mask
            mosaic_annot.extend(objects)
        return mosaic_img, mosaic_annot, mosaic_mask

    def plot_examples(
        self, idxs: list[int], nrows: int = 1, stage_idxs: list[int] = [1, 0]
    ) -> np.ndarray:
        return super().plot_examples(idxs, nrows=nrows, stage_idxs=stage_idxs)

    def plot_raw(self, idx: int, max_size: int) -> np.ndarray:
        raw_image, raw_annot, raw_mask = self.get_raw_data(idx)
        raw_joints = get_coco_joints(raw_annot)
        kpts = raw_joints[..., :2]
        visibility = raw_joints[..., 2]
        raw_image = plot_connections(raw_image.copy(), kpts, visibility, self.limbs, thr=0.5)
        overlay = raw_image.copy()

        for i, obj in enumerate(raw_annot):
            seg = obj["segmentation"]
            if isinstance(seg, dict):  # RLE annotation
                seg = coco_rle_to_seg(seg)
            c = get_color(i).tolist()
            for poly_seg in seg:
                poly_seg = np.array(poly_seg).reshape(-1, 2).astype(np.int32)
                overlay = cv2.fillPoly(overlay, [poly_seg], color=c)
                overlay = cv2.drawContours(overlay, [poly_seg], -1, color=c, thickness=1)

        alpha = 0.4  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        raw_image = cv2.addWeighted(overlay, alpha, raw_image, 1 - alpha, 0)

        raw_h, raw_w = raw_image.shape[:2]

        raw_img_transform = A.Compose(
            [
                A.LongestMaxSize(max_size),
                A.PadIfNeeded(max_size, max_size, border_mode=cv2.BORDER_CONSTANT),
            ],
        )
        raw_image = raw_img_transform(image=raw_image)["image"]
        put_txt(raw_image, ["Raw Image", f"Shape: {raw_h} x {raw_w}"])
        return raw_image

    def plot(self, idx: int, nrows: int = 3, stage_idxs: list[int] = [1]) -> np.ndarray:
        def plot_stage(stage_idx: int) -> np.ndarray:
            crowd_mask = mask_list[stage_idx]
            h, w = crowd_mask.shape[:2]
            filename = Path(self.images_filepaths[idx]).stem
            if stage_idx == 1:
                font_scale = 0.5
                labels = [
                    "Transformed Image",
                    f"Sample: {filename}",
                    f"Stage: {stage_idx} ({h} x {w})",
                ]
            else:
                font_scale = 0.25
                labels = [f"Stage: {stage_idx} ({h} x {w})"]
            image = cv2.resize(img_npy, (w, h))
            kpts_coords = joints_list[stage_idx][..., :2]
            visibility = joints_list[stage_idx][..., 2]
            kpts_heatmaps = plot_heatmaps(image, heatmaps[stage_idx])
            image = plot_connections(image.copy(), kpts_coords, visibility, self.limbs, thr=0.5)
            crowd_mask = cv2.cvtColor((crowd_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            if stage_idx == 1:
                for i in range(len(kpts_heatmaps)):
                    put_txt(kpts_heatmaps[i], [self.labels[i]], alpha=0.8, font_scale=font_scale)
                put_txt(crowd_mask, ["Crowd Mask"], alpha=1, font_scale=font_scale)
            put_txt(image, labels, alpha=0.8, font_scale=font_scale)
            grid = make_grid([image, crowd_mask, *kpts_heatmaps], nrows=nrows).astype(np.uint8)
            return grid

        img, heatmaps, mask_list, joints_list = self[idx]
        img_npy = KeypointsTransform.inverse_transform(img)
        stages_grids = [plot_stage(stage_idx) for stage_idx in stage_idxs]
        model_input_grid = make_grid(stages_grids, nrows=len(stages_grids), match_size=True).astype(
            np.uint8
        )
        raw_image = self.plot_raw(idx, max_size=model_input_grid.shape[0])
        sample_vis = stack_horizontally([raw_image, model_input_grid])
        return sample_vis

    def __getitem__(
        self, idx: int
    ) -> tuple[
        Tensor | np.ndarray,
        list[Tensor | np.ndarray],
        list[Tensor | np.ndarray],
        list[Tensor | np.ndarray],
    ]:
        if random.random() < self.mosaic_probability:
            img, annot, mask = self.get_raw_mosaiced_data(idx)
        else:
            img, annot, mask = self.get_raw_data(idx)

        annots = [obj for obj in annot if obj["iscrowd"] == 0 or obj["num_keypoints"] > 0]
        joints = get_coco_joints(annots)
        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        if self.transform is not None:
            img, mask_list, joints_list = self.transform(img, mask_list, joints_list)
        heatmaps = []
        for i in range(self.num_scales):
            joints_list[i] = self.joints_generators[i](joints_list[i])
            hms = self.hm_generators[i](joints_list[i])
            heatmaps.append(hms.astype(np.float32))
        return img, heatmaps, mask_list, joints_list

    def __len__(self) -> int:
        return len(self.annots_filepaths)


_polygons = list[list[int]]


k_i = [26, 25, 25, 35, 35, 79, 79, 72, 72, 62, 62, 107, 107, 87, 87, 89, 89]
k_i = np.array(k_i) / 1000
variances = (k_i * 2) ** 2


def object_OKS(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    obj_polygons: _polygons,
) -> float:
    # pred_kpts shape: [num_kpts, 2]
    # target_kpts shape: [num_kpts, 2]
    # 2 for: x, y
    if target_vis.sum() <= 0:
        return -1

    kpts_vis = target_vis > 0
    area = sum(
        cv2.contourArea(np.array(poly).reshape(-1, 2).astype(np.int32)) for poly in obj_polygons
    )
    area += np.spacing(1)

    dist = ((pred_kpts - target_kpts) ** 2).sum(-1)
    # dist is already squared (euclidean distance has square root)
    e = dist / (2 * variances * area)
    e = e[kpts_vis]
    num_vis_kpts = kpts_vis.sum()
    e = np.exp(-e)
    oks = np.sum(e) / num_vis_kpts
    return oks


def image_OKS(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    seg_polygons: list[_polygons],
) -> float:
    num_obj = len(target_kpts)
    oks_values = []
    for j in range(num_obj):
        dist = ((pred_kpts[j] - target_kpts[j]) ** 2).sum(-1) ** 0.5
        oks = object_OKS(pred_kpts[j], target_kpts[j], target_vis[j], seg_polygons[j])
        oks_values.append(oks)

    oks_values = np.array(oks_values).round(3)
    valid_oks_mask = oks_values != -1
    if valid_oks_mask.sum() > 0:
        oks_avg = oks_values[valid_oks_mask].mean()
        return oks_avg
    return -1


if __name__ == "__main__":
    from PIL import Image

    from src.keypoints.transforms import KeypointsTransform

    out_size = 512
    hm_resolutions = [1 / 4, 1 / 2]
    transform = KeypointsTransform(out_size, hm_resolutions, min_scale=0.25, max_scale=1.25)
    # ds = CocoKeypointsDataset("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = CocoKeypointsDataset(
        "data/COCO/raw",
        "train2017",
        transform.train,
        out_size,
        hm_resolutions,
        mosaic_probability=0.25,
    )
    grid = ds.plot_examples([0, 1, 2, 3, 4, 5, 6], nrows=1, stage_idxs=[1, 0])
    Image.fromarray(grid).save("test.jpg")
    # ds.explore(0)
