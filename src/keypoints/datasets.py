import numpy as np
from torch import Tensor
import torch
import cv2

from src.utils.files import load_yaml
from src.utils.image import make_grid, put_txt
from src.base.datasets import BaseImageDataset
import torchvision.transforms.functional as F


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

    def __init__(self, output_size: tuple[int, int], sigma: float = 7):
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


class BaseKeypointsDataset(BaseImageDataset):
    transform: KeypointsTransform

    def __init__(
        self,
        root: str,
        split: str,
        transform: KeypointsTransform,
        hm_resolutions: list[float],
        labels: list[str],
        limbs: list[tuple[int, int]],
    ):
        super().__init__(root, split, transform)
        self.labels = labels
        self.limbs = limbs
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

    def __len__(self) -> int:
        return len(self.annots_filepaths)

    def load_annot(self, idx: int):
        return load_yaml(self.annots_filepaths[idx])

    def _transform(
        self,
        image: np.ndarray,
        keypoints: list[tuple[int, int]],
        visibilities: list[int],
        num_obj: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.is_train:
            transform = self.transform.random
        else:
            transform = self.transform.inference

        transformed = transform(
            image=image, keypoints=keypoints, visibilities=visibilities
        )
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        visibilities = transformed["visibilities"]

        transformed = self.transform.preprocessing(
            image=image, keypoints=keypoints, visibilities=visibilities
        )
        tr_image = transformed["image"]
        tr_keypoints = np.array(transformed["keypoints"]).astype(np.int32)
        tr_visibilities = np.array(transformed["visibilities"])

        transformed = self.transform.postprocessing(
            image=tr_image, keypoints=tr_keypoints, visibilities=tr_visibilities
        )
        tr_image = transformed["image"]
        tr_keypoints = np.array(transformed["keypoints"]).astype(np.int32)
        tr_visibilities = np.array(transformed["visibilities"])
        tr_keypoints = tr_keypoints.reshape(num_obj, -1, 2)
        tr_visibilities = tr_visibilities.reshape(num_obj, -1)
        return tr_image, tr_keypoints, tr_visibilities

    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int], int]:
        raise NotImplementedError()

    def plot(self, idx: int, hm_idx: int = 0):
        raw_image = self.load_image(idx)
        image, all_heatmaps, target_weights, keypoints, visibilities = self[idx]

        heatmaps = all_heatmaps[hm_idx]
        h, w = image.shape[-2:]
        heatmaps = F.resize(torch.from_numpy(heatmaps), [h, w])

        image = self.transform.inverse_preprocessing(image)

        raw_size = max(*raw_image.shape[:2])
        ymin, xmin = [(raw_size - size) // 2 for size in raw_image.shape[:2]]
        blank = np.zeros((raw_size, raw_size, 3), dtype=np.uint8)
        if ymin > 0:  # pad y
            ymax = raw_image.shape[0] + ymin
            blank[ymin:ymax] = raw_image
        else:  # pad x
            xmax = raw_image.shape[1] + xmin
            blank[:, xmin:xmax] = raw_image
        tr_h, tr_w = image.shape[:2]

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
        kpts_heatmaps.insert(0, image)

        images = [raw_image]
        for kpt_heatmap, label in zip(kpts_heatmaps[1:], self.labels):
            put_txt(kpt_heatmap, [label])
        images.extend(kpts_heatmaps)

        hms_grid = make_grid(images, nrows=3, resize=0.5)
        img_txt = self.images_filepaths[idx].split("/")[-1] + f" ({idx}/{len(self)})"
        put_txt(hms_grid, [img_txt])
        return hms_grid

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray,]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)

        keypoints, visibilities, num_obj = self.parse_annot(annot)
        image, keypoints, visibilities = self._transform(
            image, keypoints, visibilities, num_obj
        )

        max_h, max_w = image.shape[-2:]

        scales_heatmaps = []
        for hm_generator in self.hm_generators:
            heatmaps, target_weights = hm_generator(
                keypoints, visibilities, max_h, max_w
            )
            scales_heatmaps.append(heatmaps)

        return image, scales_heatmaps, target_weights, keypoints, visibilities


def mppe_collate_fn(
    batch: tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]
) -> tuple[
    Tensor,
    list[Tensor],
    Tensor,
    list[list[list[tuple[int, int]]]],
    list[list[list[float]]],
]:
    images = [item[0] for item in batch]
    scales_heatmaps = [item[1] for item in batch]
    target_weights = [item[2] for item in batch]
    keypoints = [item[3].tolist() for item in batch]
    visibilities = [item[4].tolist() for item in batch]

    images = torch.from_numpy(np.stack(images))
    scales_heatmaps = torch.from_numpy(np.stack(scales_heatmaps))
    scales_heatmaps = [scales_heatmaps[:, i] for i in range(scales_heatmaps.shape[1])]
    target_weights = torch.from_numpy(np.stack(target_weights))

    return images, scales_heatmaps, target_weights, keypoints, visibilities


class MPPEKeypointsDataset(BaseKeypointsDataset):
    transform: MPPEKeypointsTransform

    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int], int]:
        objects_annot = annot["objects"]
        num_objects = len(objects_annot)
        keypoints = []
        visibilities = []
        for obj_annots in objects_annot:
            kpts = obj_annots["keypoints"]
            for kpt in kpts:
                x, y = kpt["x"], kpt["y"]
                visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
                keypoints.append([x, y])
                visibilities.append(visibility)

        return keypoints, visibilities, num_objects


class SPPEKeypointsDataset(BaseKeypointsDataset):
    transform: SPPEKeypointsTransform

    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int], int]:
        keypoints = []
        visibilities = []
        num_objects = 1

        kpts = annot["keypoints"]
        for kpt in kpts:
            x, y = kpt["x"], kpt["y"]
            visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
            keypoints.append([x, y])
            visibilities.append(visibility)
        return keypoints, visibilities, num_objects


if __name__ == "__main__":
    from src.utils.config import DS_ROOT

    mode = "MPPE"
    ds_name = "COCO"
    split = "train"

    if ds_name == "MPII":
        from geda.data_providers.mpii import LABELS, LIMBS
    else:
        from geda.data_providers.coco import LABELS, LIMBS

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "SPPE":
        ds_subdir = "SPPEHumanPose"
        out_size = (256, 192)
        transform = SPPEKeypointsTransform(mean=mean, std=std, out_size=out_size)
        Dataset = SPPEKeypointsDataset

    else:
        ds_subdir = "HumanPose"
        out_size = (512, 512)
        transform = MPPEKeypointsTransform(mean=mean, std=std, out_size=out_size)
        Dataset = MPPEKeypointsDataset

    ds_root = str(DS_ROOT / ds_name / ds_subdir)

    hm_resolutions = [1 / 2, 1 / 4]

    ds = Dataset(ds_root, split, transform, hm_resolutions, labels=LABELS, limbs=LIMBS)
    ds.explore(idx=0)
