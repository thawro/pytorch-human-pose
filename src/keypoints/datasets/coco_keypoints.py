import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pycocotools
import torch
from geda.data_providers.coco import LABELS as coco_labels
from geda.data_providers.coco import LIMBS as coco_limbs
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.keypoints.datasets.transforms import ComposeKeypointsTransform, KeypointsTransform
from src.keypoints.visualization import plot_connections, plot_heatmaps
from src.logger.pylogger import log
from src.utils.files import load_yaml, save_yaml
from src.utils.image import make_grid
from src.utils.utils import get_rank


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


class CocoKeypoints(BaseImageDataset):
    limbs = coco_limbs
    labels = coco_labels

    def __init__(
        self,
        name: str,
        root: str,
        split: str,
        transform: ComposeKeypointsTransform,
        out_size: int = 512,
        hm_resolutions: list[float] = [1 / 4, 1 / 2],
    ):
        self.name = name
        self.root = root
        self.split = split
        kpts_dir = f"person_keypoints_{self.split}"
        self.images_dir = f"{self.root}/images/{self.split}"
        self.annots_dir = f"{self.root}/annotations/{kpts_dir}"
        self.masks_dir = f"{self.root}/masks/{kpts_dir}"
        self.out_size = out_size
        self.is_train = "train" in split
        self.num_scales = len(hm_resolutions)
        self.num_kpts = 17
        self.max_num_people = 30
        self._save_annots_to_files()
        self._set_paths()
        self.transform = transform
        self.hm_resolutions = hm_resolutions
        self.hm_sizes = [int(res * out_size) for res in hm_resolutions]
        hm_generators = []
        joints_generator = []
        for hm_size in self.hm_sizes:
            hm_generators.append(HeatmapGenerator(self.num_kpts, hm_size, sigma=2))
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
        log.info(f"..Saving {self.split} annotations (keypoints and crowd masks) to files..")
        rank = get_rank()
        if rank != 0:
            log.warn(
                f"     Current process (rank = {rank}) is not the main process (rank = 0) -> Skipping"
            )
            return
        if os.path.exists(self.annots_dir):
            log.info(f"     {self.split} annotations already saved to files -> Skipping")
            return
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

    def get_raw_data(self, idx: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
        image = np.array(Image.open(self.images_filepaths[idx]).convert("RGB"))
        annot = load_yaml(self.annots_filepaths[idx])
        mask = np.load(self.masks_filepaths[idx])
        return image, annot, mask

    def get_joints(self, annots: list[dict]):
        num_people = len(annots)
        joints = np.zeros((num_people, self.num_kpts, 3))
        for i, obj in enumerate(annots):
            joints[i] = np.array(obj["keypoints"]).reshape([-1, 3])
        return joints

    def plot(self, idx: int) -> np.ndarray:
        img, heatmaps, mask_list, joints_list = self[idx]
        h, w = mask_list[1].shape[:2]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img.permute(1, 2, 0).numpy() * std) + mean
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (h, w))
        kpts_coords = joints_list[1][..., :2]
        visibility = joints_list[1][..., 2]
        kpts_heatmaps = plot_heatmaps(img, heatmaps[1])
        img = plot_connections(img.copy(), kpts_coords, visibility, self.limbs, thr=0.5)
        mask = cv2.cvtColor((mask_list[1] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        grid = make_grid([img, mask, *kpts_heatmaps]).astype(np.uint8)
        return grid

    def __getitem__(self, idx):
        img, annot, mask = self.get_raw_data(idx)
        annots = [obj for obj in annot if obj["iscrowd"] == 0 or obj["num_keypoints"] > 0]
        joints = self.get_joints(annots)
        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        img, mask_list, joints_list = self.transform(img, mask_list, joints_list)
        heatmaps = []
        for i in range(self.num_scales):
            joints_list[i] = self.joints_generators[i](joints_list[i])
            hms = self.hm_generators[i](joints_list[i])
            heatmaps.append(hms.astype(np.float32))
        return img, heatmaps, mask_list, joints_list

    def __len__(self) -> int:
        return len(self.annots_filepaths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from src.keypoints.datasets.transforms import KeypointsTransform

    out_size = 512
    hm_resolutions = [1 / 4, 1 / 2]
    transform = KeypointsTransform(out_size, hm_resolutions)
    # ds = CocoKeypoints("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = CocoKeypoints("data/COCO/raw", "train2017", transform.train, out_size, hm_resolutions)
    ds.explore(0)
