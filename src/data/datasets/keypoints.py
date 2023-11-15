from src.data.datasets.base import BaseImageDataset
from src.data.transforms.keypoints import KeypointsTransform, KeypointsImageTransform
import numpy as np
from torch import Tensor
from src.utils.files import load_yaml
import torch
from src.visualization.keypoints import plot_single_image
import matplotlib.pyplot as plt
import albumentations as A


class HeatmapGenerator:
    """
    source: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
    """

    def __init__(self, output_size: tuple[int, int], sigma: float = 7):
        self.output_size = output_size
        self.h, self.w = output_size
        if sigma < 0:
            sigma = max(output_size) / 32
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
    transform: KeypointsImageTransform

    def __init__(
        self,
        root: str,
        split: str,
        transform: KeypointsImageTransform,
        labels: list[str],
    ):
        super().__init__(root, split, transform)
        self.labels = labels
        self.annots_filepaths = [
            path.replace(".jpg", ".yaml").replace("images", "annots")
            for path in self.images_filepaths
        ]
        self.size = transform.size
        self.hm_size = transform.size // 4
        self.heatmap_generator = HeatmapGenerator(
            (self.hm_size, self.hm_size), sigma=-1
        )

    def __len__(self) -> int:
        return len(self.annots_filepaths)

    def load_annot(self, idx: int):
        return load_yaml(self.annots_filepaths[idx])

    def parse_annot(self, annot: dict) -> tuple[list[tuple[int, int]], list[int]]:
        objects = annot["objects"]
        keypoints = []
        visibilities = []
        for obj_annots in objects:
            kpts = obj_annots["keypoints"]
            for kpt in kpts:
                keypoints.append([kpt["x"], kpt["y"]])
                visibilities.append(kpt["visibility"])
        return keypoints, visibilities

    def _transform(
        self,
        image: np.ndarray,
        keypoints: list[tuple[int, int]],
        visibilities: list[int],
        num_obj: int,
    ) -> tuple[Tensor, list[list[tuple[int, int]]], list[list[int]]]:
        transformed = self.transform(
            image=image, keypoints=keypoints, visibilities=visibilities
        )
        tr_image = transformed["image"]
        tr_keypoints = np.array(transformed["keypoints"]).astype(np.int32)
        tr_visibilities = np.array(transformed["visibilities"])
        tr_keypoints = tr_keypoints.reshape(num_obj, -1, 2).tolist()
        tr_visibilities = tr_visibilities.reshape(num_obj, -1).tolist()
        return tr_image, tr_keypoints, tr_visibilities


class MultiObjectsKeypointsDataset(BaseKeypointsDataset):
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
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        img_h, img_w = image.shape[:2]

        obj_sizes = []
        for obj in annot["objects"]:
            x, y, w, h = obj["bbox"]
            obj_sizes.append(w * h)
        biggest_obj_idx = np.array(obj_sizes).argmax()
        annot["objects"] = [annot["objects"][biggest_obj_idx]]
        keypoints, visibilities = self.parse_annot(annot)

        x, y, w, h = annot["objects"][0]["bbox"]
        # COCO
        xmin, xmax = int(x), int(x + w)
        ymin, ymax = int(y), int(y + h)

        # MPII #TODO: change in geda
        # xmin, xmax = int(x - w // 2), int(x + w // 2)
        # ymin, ymax = int(y - h // 2), int(y + h // 2)

        xmin, xmax = max(0, xmin), min(xmax, img_w)
        ymin, ymax = max(0, ymin), min(ymax, img_h)

        crop = A.Compose(
            [A.Crop(xmin, ymin, xmax, ymax)],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["visibilities"], remove_invisible=False
            ),
        )
        cropped = crop(image=image, keypoints=keypoints, visibilities=visibilities)
        image = cropped["image"]
        keypoints = cropped["keypoints"]
        visibilities = cropped["visibilities"]

        image, keypoints, visibilities = self._transform(
            image, keypoints, visibilities, num_obj=1
        )
        heatmaps, target_weights = self.heatmap_generator(
            keypoints, visibilities, self.size, self.size
        )
        heatmaps = torch.from_numpy(heatmaps)
        target_weights = torch.from_numpy(target_weights)

        return image, heatmaps, target_weights


def plot(image: Tensor, heatmaps: Tensor, transform):
    hms = heatmaps.cpu().numpy()
    img = transform.inverse_preprocessing(image)
    hms_grid = plot_single_image(img, hms)
    plt.figure(figsize=(24, 12))
    plt.imshow(hms_grid)
    plt.savefig("example.jpg")


if __name__ == "__main__":
    from src.utils.config import DS_ROOT
    from geda.data_providers.coco import LABELS

    ds_root = str(DS_ROOT / "COCO" / "HumanPose")
    transform = KeypointsTransform(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=256
    )
    ds = SingleObjectKeypointsDataset(
        ds_root, "val", transform=transform.inference, labels=LABELS
    )
    image, heatmaps, weights = ds[22]
    plot(image, heatmaps, ds.transform)
