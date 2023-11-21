import numpy as np
from torch import Tensor
import torch
import cv2

from src.utils.files import load_yaml
from src.utils.image import make_grid, put_txt
from src.base.datasets import BaseImageDataset

from src.keypoints.transforms import KeypointsTransform, SPPEKeypointsTransform
from src.keypoints.visualization import create_heatmaps


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
        joints_centers is a dict in form:
        {
            joint_id: [(x1, y1, v1), (x2, y2, v2), ..., (xn, yn, vn)],
            ...
        }
        xy - coords,
        v - visibility
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
        output_sizes: list[tuple[int, int]],
        labels: list[str],
        limbs: list[tuple[int, int]],
    ):
        super().__init__(root, split, transform)
        self.labels = labels
        self.limbs = limbs
        self.output_sizes = output_sizes
        self.is_train = split == "train"
        self.annots_filepaths = [
            path.replace(".jpg", ".yaml").replace("images", "annots")
            for path in self.images_filepaths
        ]
        self.size = transform.size
        hm_generators = []
        for output_size in output_sizes:
            hm_generators.append(HeatmapGenerator(output_size, sigma=2))
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
    ) -> tuple[Tensor, list[list[tuple[int, int]]], list[list[int]]]:
        image = self.transform.preprocessing(image=image)["image"]

        transformed = self.transform.postprocessing(
            image=image, keypoints=keypoints, visibilities=visibilities
        )
        tr_image = transformed["image"]
        tr_keypoints = np.array(transformed["keypoints"]).astype(np.int32)
        tr_visibilities = np.array(transformed["visibilities"])
        tr_keypoints = tr_keypoints.reshape(num_obj, -1, 2).tolist()
        tr_visibilities = tr_visibilities.reshape(num_obj, -1).tolist()
        return tr_image, tr_keypoints, tr_visibilities

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()

    def plot(self, idx: int, hm_idx: int = 1):
        raw_image = self.load_image(idx)
        image, all_heatmaps, target_weights = self[idx]

        heatmaps = all_heatmaps[hm_idx]
        hms = heatmaps
        # hms = heatmaps.cpu().numpy()
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

        kpts_heatmaps = create_heatmaps(image, hms, limbs=self.limbs)

        images = [raw_image]
        for kpt_heatmap, label in zip(kpts_heatmaps[1:], self.labels):
            put_txt(kpt_heatmap, [label])
        images.extend(kpts_heatmaps)

        hms_grid = make_grid(images, nrows=3)
        img_txt = self.images_filepaths[idx].split("/")[-1] + f" ({idx}/{len(self)})"
        put_txt(hms_grid, [img_txt])
        return hms_grid


class MultiObjectsKeypointsDataset(BaseKeypointsDataset):
    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int]]:
        objects = annot["objects"]
        keypoints = []
        visibilities = []
        for obj_annots in objects:
            kpts = obj_annots["keypoints"]
            for kpt in kpts:
                x, y = kpt["x"], kpt["y"]
                visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
                keypoints.append([x, y])
                visibilities.append(visibility)
        return keypoints, visibilities

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        num_obj = len(annot["objects"])
        keypoints, visibilities = self.parse_annot(annot)
        image, keypoints, visibilities = self._transform(
            image, keypoints, visibilities, num_obj
        )
        heatmaps, target_weights = self.heatmap_generator(
            keypoints, visibilities, self.size, self.size
        )
        heatmaps = torch.from_numpy(heatmaps)
        target_weights = torch.from_numpy(target_weights)

        return image, heatmaps, target_weights


class SingleObjectKeypointsDataset(BaseKeypointsDataset):
    transform: SPPEKeypointsTransform

    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int]]:
        keypoints = []
        visibilities = []

        kpts = annot["keypoints"]
        for kpt in kpts:
            x, y = kpt["x"], kpt["y"]
            visibility = int((x > 0 and y > 0) or kpt["visibility"] > 0)
            keypoints.append([x, y])
            visibilities.append(visibility)
        return keypoints, visibilities

    def __getitem__(self, idx: int) -> tuple[Tensor, list[np.ndarray], Tensor]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        img_h, img_w = image.shape[:2]

        keypoints, visibilities = self.parse_annot(annot)

        if self.is_train:
            transformed = self.transform.random(
                image=image, keypoints=keypoints, visibilities=visibilities
            )
            image = transformed["image"]
            keypoints = transformed["keypoints"]
            visibilities = transformed["visibilities"]

        image, keypoints, visibilities = self._transform(
            image, keypoints, visibilities, num_obj=1
        )
        all_heatmaps = []
        max_h, max_w = self.size, self.size
        for hm_generator in self.hm_generators:
            heatmaps, target_weights = hm_generator(
                keypoints, visibilities, max_h, max_w
            )
            all_heatmaps.append(heatmaps)
        target_weights = torch.from_numpy(target_weights)

        return image, all_heatmaps, target_weights


if __name__ == "__main__":
    from src.utils.config import DS_ROOT

    ds_name = "MPII"
    split = "train"

    if ds_name == "MPII":
        from geda.data_providers.mpii import LABELS, LIMBS
    else:
        from geda.data_providers.coco import LABELS, LIMBS

    ds_root = str(DS_ROOT / ds_name / "SPPEHumanPose")
    transform = SPPEKeypointsTransform(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=256
    )
    output_sizes = [(128, 128), (64, 64)]
    # output_sizes = [(128, 96), (64, 48)]

    ds = SingleObjectKeypointsDataset(
        ds_root,
        split,
        output_sizes=output_sizes,
        transform=transform,
        labels=LABELS,
        limbs=LIMBS,
    )
    # ds = MultiObjectsKeypointsDataset(
    #     ds_root, "val", transform=transform.inference, labels=LABELS
    # )
    ds.explore(idx=0)
