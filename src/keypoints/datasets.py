import numpy as np
from torch import Tensor
import torch
import cv2
from src.utils.files import load_yaml
from src.utils.image import make_grid, put_txt, get_color
from src.base.datasets import BaseImageDataset
from src.keypoints.utils import (
    xyxy_to_mask,
    mask_to_head_xyxy,
    polygons_to_mask,
    mask_to_polygons,
)
from typing import Callable
import torchvision.transforms.functional as F
from abc import abstractmethod
from geda.data_providers.coco import LABELS as coco_labels, LIMBS as coco_limbs
from geda.data_providers.mpii import LABELS as mpii_labels, LIMBS as mpii_limbs


from src.keypoints.transforms import (
    KeypointsTransform,
    SPPEKeypointsTransform,
    MPPEKeypointsTransform,
)


from src.keypoints.visualization import plot_heatmaps, plot_connections


class HeatmapGenerator:
    """
    source: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
    """

    def __init__(self, output_size: tuple[int, int], sigma: float = 2):
        self.output_size = output_size
        self.h, self.w = output_size
        if sigma < 0:
            sigma = max(output_size) / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(
        self,
        keypoints: list[list[tuple[int, int]]],
        visibilities: list[list[int]],
        max_h: int,
        max_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        visibility: 0 - not labeled, 1 - labeled but not visible, 2 - labeled and visible
        """
        num_objects = len(keypoints)
        num_kpts = len(keypoints[0])
        hms = np.zeros((num_kpts, self.h, self.w), dtype=np.float32)
        target_weights = np.zeros(num_kpts, dtype=np.float32)
        for kpts, vis in zip(keypoints, visibilities):
            for idx in range(len(kpts)):
                x, y = kpts[idx]
                x = int((x / max_w) * self.w)
                y = int((y / max_h) * self.h)

                v = vis[idx]

                if v > 0:
                    if x < 0 or y < 0 or x >= self.w or y >= self.h:
                        continue
                    xmin = int(np.round(x - 3 * self.sigma - 1))
                    ymin = int(np.round(y - 3 * self.sigma - 1))
                    xmax = int(np.round(x + 3 * self.sigma + 2))
                    ymax = int(np.round(y + 3 * self.sigma + 2))

                    c, d = max(0, -xmin), min(xmax, self.w) - xmin
                    a, b = max(0, -ymin), min(ymax, self.h) - ymin

                    cc, dd = max(0, xmin), min(xmax, self.w)
                    aa, bb = max(0, ymin), min(ymax, self.h)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.gauss[a:b, c:d]
                    )
                    target_weights[idx] = 1
        return hms, target_weights


class MPIIDataset:
    fn_coords2masks = xyxy_to_mask
    labels = mpii_labels
    limbs = mpii_limbs

    @classmethod
    def fn_masks2coords(cls, mask: np.ndarray):
        return [mask_to_head_xyxy(mask)]

    def plot_extra_coords(self, image: np.ndarray, heads_bboxes: list[list[int]]):
        for head_bbox in heads_bboxes:
            xmin, ymin, xmax, ymax = head_bbox[0]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (100, 255, 100), 2)
        return image


class COCODataset:
    fn_coords2masks = polygons_to_mask
    fn_masks2coords = mask_to_polygons
    labels = coco_labels
    limbs = coco_limbs

    def plot_extra_coords(self, image: np.ndarray, objects_polygons) -> np.ndarray:
        h, w = image.shape[:2]

        seg_masks = [polygons_to_mask(polygons, h, w) for polygons in objects_polygons]
        for i, mask in enumerate(seg_masks):
            _mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * get_color(i)
            image = cv2.addWeighted(image, 1, _mask, 1, 0)
        return image


class BaseKeypointsDataset(BaseImageDataset):
    transform: KeypointsTransform
    is_multiobj: bool
    fn_coords2masks: Callable
    fn_masks2coords: Callable
    labels: list[str]
    limbs: list[tuple[int, int]]

    def __init__(
        self,
        root: str,
        split: str,
        transform: KeypointsTransform,
        hm_resolutions: list[float],
    ):
        super().__init__(root, split, transform)
        out_size = transform.out_size
        self.hm_resolutions = hm_resolutions
        self.hm_sizes = [
            (int(res * out_size[0]), int(res * out_size[1])) for res in hm_resolutions
        ]
        self.is_train = split == "train"
        self.annots_filepaths = [
            path.replace(".jpg", ".yaml").replace("images", "annots")
            for path in self.images_filepaths
        ]
        hm_generators = []
        for hm_size in self.hm_sizes:
            hm_generators.append(HeatmapGenerator(hm_size, sigma=2))
        self.hm_generators = hm_generators

    @property
    def is_mpii(self) -> bool:
        return self.name == "MPII"

    @property
    def name(self) -> str:
        if "COCO" in str(self.root):
            return "COCO"
        return "MPII"

    def __len__(self) -> int:
        return len(self.annots_filepaths)

    def load_annot(self, idx: int):
        return load_yaml(self.annots_filepaths[idx])

    def extra_coords_to_masks(self, extra_coords, h: int, w: int) -> list[np.ndarray]:
        return [self.__class__.fn_coords2masks(coords, h, w) for coords in extra_coords]

    def masks_to_extra_coords(self, masks: list[np.ndarray]):
        return [self.__class__.fn_masks2coords(mask) for mask in masks]

    def _transform(
        self,
        image: np.ndarray,
        keypoints: list[tuple[int, int]],
        visibilities: list[int],
        masks: list[np.ndarray],
        num_obj: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        if self.is_train:
            transform = self.transform.random
        else:
            transform = self.transform.inference

        transformed = transform(
            image=image, keypoints=keypoints, visibilities=visibilities, masks=masks
        )
        transformed = self.transform.preprocessing(**transformed)
        transformed = self.transform.postprocessing(**transformed)
        _image = transformed["image"]
        _masks = transformed["masks"]
        _keypoints = np.array(transformed["keypoints"]).astype(np.int32)
        _visibilities = np.array(transformed["visibilities"])
        _keypoints = _keypoints.reshape(num_obj, -1, 2)
        _visibilities = _visibilities.reshape(num_obj, -1)
        return _image, _keypoints, _visibilities, _masks

    @abstractmethod
    def parse_annot(
        self, annot: dict
    ) -> tuple[list[tuple[int, int]], list[int], int, list[list[list[int]]]]:
        raise NotImplementedError()

    @abstractmethod
    def get_extras_from_annot(self, annot: dict) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def plot_raw_annot_coords(self, image: np.ndarray, raw_annot) -> np.ndarray:
        raise NotImplementedError()

    def plot(self, idx: int, hm_idx: int = 0):
        raw_image, raw_annot = self.get_raw_data(idx)
        image, all_heatmaps, _, keypoints, visibilities, extra_coords = self[idx]

        heatmaps = all_heatmaps[hm_idx]

        image = self.transform.inverse_preprocessing(image)
        tr_h, tr_w = image.shape[:2]

        heatmaps = F.resize(torch.from_numpy(heatmaps), [tr_h, tr_w])

        raw_image = self.plot_raw_annot_coords(raw_image, raw_annot)

        raw_size = max(*raw_image.shape[:2])
        ymin, xmin = [(raw_size - size) // 2 for size in raw_image.shape[:2]]
        blank = np.zeros((raw_size, raw_size, 3), dtype=np.uint8)
        if ymin > 0:  # pad y
            ymax = raw_image.shape[0] + ymin
            blank[ymin:ymax] = raw_image
        else:  # pad x
            xmax = raw_image.shape[1] + xmin
            blank[:, xmin:xmax] = raw_image

        raw_image = cv2.resize(blank, (tr_w, tr_h))

        kpts_heatmaps = plot_heatmaps(image, heatmaps.numpy())

        scores = []
        for i in range(len(keypoints)):
            obj_scores = []
            for j in range(len(keypoints[i])):
                score = (visibilities[i][j] > 0) * 1
                obj_scores.append(score)
            scores.append(obj_scores)

        image = plot_connections(image.copy(), keypoints, scores, self.limbs, thr=0.5)

        image = self.plot_extra_coords(image, extra_coords)

        kpts_heatmaps.insert(0, image)

        images = [raw_image]
        for kpt_heatmap, label in zip(kpts_heatmaps[1:], self.labels):
            put_txt(kpt_heatmap, [label])
        images.extend(kpts_heatmaps)

        hms_grid = make_grid(images, nrows=3, resize=0.5)
        img_txt = self.images_filepaths[idx].split("/")[-1] + f" ({idx}/{len(self)})"
        put_txt(hms_grid, [img_txt], font_scale=0.25)
        hms_grid = cv2.resize(hms_grid, dsize=(0, 0), fx=2, fy=2)
        return hms_grid

    def get_raw_data(self, idx) -> tuple[np.ndarray, dict]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        return image, annot

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[int]],
    ]:
        image, annot = self.get_raw_data(idx)
        h, w = image.shape[:2]
        keypoints, visibilities, num_obj, extra_coords = self.parse_annot(annot)
        masks = self.extra_coords_to_masks(extra_coords, h, w)
        image, keypoints, visibilities, masks = self._transform(
            image, keypoints, visibilities, masks, num_obj
        )
        masks = [mask.numpy() for mask in masks]
        extra_coords = self.masks_to_extra_coords(masks)

        max_h, max_w = image.shape[-2:]
        scales_heatmaps = []
        for hm_generator in self.hm_generators:
            heatmaps, target_weights = hm_generator(
                keypoints, visibilities, max_h, max_w
            )
            scales_heatmaps.append(heatmaps)

        return (
            image,
            scales_heatmaps,
            target_weights,
            keypoints,
            visibilities,
            extra_coords,
        )


def collate_fn(
    batch: tuple[
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[list[list[int]]]],
    ]
) -> tuple[
    Tensor,
    list[Tensor],
    Tensor,
    list[list[list[tuple[int, int]]]],
    list[list[list[float]]],
    list[list[list[list[int]]]],
]:
    # extra_coords shape: [batch_size, num_obj, num_polygons, num_coords*2]
    images = [item[0] for item in batch]
    batch_size = len(images)
    scales_heatmaps = [item[1] for item in batch]
    target_weights = [item[2] for item in batch]
    target_keypoints = [item[3] for item in batch]
    target_visibilities = [item[4] for item in batch]
    extra_coords = [item[5] for item in batch]

    images = torch.from_numpy(np.stack(images))
    # scales_heatmaps = torch.from_numpy(np.stack(scales_heatmaps))

    num_resolutions = len(scales_heatmaps[0])
    tensor_scales_heatmaps = []
    # num_scales - element list with elements of shape
    # batch_size, num_kpts, scale_h, scale_w
    for i in range(num_resolutions):
        resolution_heatmaps = [scales_heatmaps[b][i] for b in range(batch_size)]
        resolution_heatmaps = torch.from_numpy(np.stack(resolution_heatmaps))
        tensor_scales_heatmaps.append(resolution_heatmaps)
    target_weights = torch.from_numpy(np.stack(target_weights))

    return (
        images,
        tensor_scales_heatmaps,
        target_weights,
        target_keypoints,
        target_visibilities,
        extra_coords,
    )


class SPPEKeypointsDataset(BaseKeypointsDataset):
    transform: SPPEKeypointsTransform
    is_multiobj: bool = False

    def parse_annot(
        self, annot: dict
    ) -> tuple[list[tuple[int, int]], list[int], int, list[list[list[int]]]]:
        num_objects = 1
        keypoints = []
        visibilities = []
        extra_coords = self.get_extras_from_annot(annot)
        kpts = annot["keypoints"]
        for kpt in kpts:
            x, y = kpt["x"], kpt["y"]
            visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
            keypoints.append([x, y])
            visibilities.append(visibility)
        return keypoints, visibilities, num_objects, extra_coords


class MPPEKeypointsDataset(BaseKeypointsDataset):
    transform: MPPEKeypointsTransform
    is_multiobj: bool = True

    def parse_annot(
        self, annot: dict
    ) -> tuple[list[tuple[int, int]], list[int], int, list[list[list[int]]]]:
        objects_annot = annot["objects"]
        extra_coords = self.get_extras_from_annot(annot)
        num_objects = len(objects_annot)
        keypoints = []
        visibilities = []
        for obj in objects_annot:
            kpts = obj["keypoints"]
            for kpt in kpts:
                x, y = kpt["x"], kpt["y"]
                visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
                keypoints.append([x, y])
                visibilities.append(visibility)

        return keypoints, visibilities, num_objects, extra_coords


class SppeMpiiDataset(SPPEKeypointsDataset, MPIIDataset):
    def get_extras_from_annot(self, annot: dict):
        return [[annot["head_xyxy"]]]

    def plot_raw_annot_coords(self, image: np.ndarray, raw_annot: dict) -> np.ndarray:
        raw_coords = self.get_extras_from_annot(raw_annot)
        return self.plot_extra_coords(image, raw_coords)


class MppeMpiiDataset(MPPEKeypointsDataset, MPIIDataset):
    def get_extras_from_annot(self, annot: dict):
        return [[obj["head_xyxy"]] for obj in annot["objects"]]

    def plot_raw_annot_coords(self, image: np.ndarray, raw_annot: dict) -> np.ndarray:
        raw_coords = self.get_extras_from_annot(raw_annot)
        return self.plot_extra_coords(image, raw_coords)


class SppeCocoDataset(SPPEKeypointsDataset, COCODataset):
    def get_extras_from_annot(self, annot: dict):
        return [annot["segmentation"]]

    def plot_raw_annot_coords(self, image: np.ndarray, raw_annot: dict) -> np.ndarray:
        raw_coords = self.get_extras_from_annot(raw_annot)
        return self.plot_extra_coords(image, raw_coords)


class MppeCocoDataset(MPPEKeypointsDataset, COCODataset):
    def get_extras_from_annot(self, annot: dict):
        return [obj["segmentation"] for obj in annot["objects"]]

    def plot_raw_annot_coords(self, image: np.ndarray, raw_annot: dict) -> np.ndarray:
        raw_coords = self.get_extras_from_annot(raw_annot)
        return self.plot_extra_coords(image, raw_coords)


if __name__ == "__main__":
    from src.utils.config import DS_ROOT

    Datasets = {
        "SPPE": {"COCO": SppeCocoDataset, "MPII": SppeMpiiDataset},
        "MPPE": {"COCO": MppeCocoDataset, "MPII": MppeMpiiDataset},
    }
    mode = "SPPE"
    mode = "MPPE"

    # ds_name = "MPII"
    ds_name = "COCO"

    Dataset = Datasets[mode][ds_name]
    split = "val"
    split = "train"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "SPPE":
        ds_subdir = "SPPEHumanPose"
        out_size = (256, 256)
        transform = SPPEKeypointsTransform(mean=mean, std=std, out_size=out_size)

    else:
        ds_subdir = "HumanPose"
        out_size = (512, 512)
        transform = MPPEKeypointsTransform(mean=mean, std=std, out_size=out_size)

    ds_root = str(DS_ROOT / ds_name / ds_subdir)

    hm_resolutions = [1 / 2, 1 / 4]

    ds = Dataset(ds_root, split, transform, hm_resolutions)
    ds.explore(idx=0, hm_idx=0)
