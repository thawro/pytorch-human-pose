from src.base.datasets import BaseImageDataset
from src.classification.transforms import ClassificationTransform
from src.utils.image import make_grid
import numpy as np
import cv2


class BaseKeypointsDataset(BaseImageDataset):
    transform: ClassificationTransform
    labels: list[str]

    def __init__(self, root: str, split: str, transform: ClassificationTransform):
        super().__init__(root, split, transform)
        self.out_size = transform.out_size
        self.is_train = split == "train"

    def __len__(self) -> int:
        return len(self.annots_filepaths)

    def load_annot(self, idx: int):
        annot = load_yaml(self.annots_filepaths[idx])
        self.raw_annots_cache[idx] = annot
        return annot

    def _transform(self, image: np.ndarray) -> np.ndarray:
        if self.is_train:
            transform = self.transform.random
        else:
            transform = self.transform.inference
        transformed = transform(image)
        transformed = self.transform.preprocessing(**transformed)
        transformed = self.transform.postprocessing(**transformed)
        return transformed["image"]

    def plot(self, idx: int):
        raw_image, raw_annot = self.get_raw_data(idx)

        image = self[idx]
        image = self.transform.inverse_preprocessing(image)
        tr_h, tr_w = image.shape[:2]

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

        grid = make_grid([raw_image, image], nrows=3, resize=0.5)
        f_xy = 1.2
        grid = cv2.resize(grid, dsize=(0, 0), fx=f_xy, fy=f_xy)
        return grid

    def get_raw_data(self, idx) -> tuple[np.ndarray, dict]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        return image, annot

    def __getitem__(self, idx: int) -> np.ndarray:
        image, annot = self.get_raw_data(idx)
        image = self._transform(image)
        return image


if __name__ == "__main__":
    from src.classification.bin.config import create_config

    cfg = create_config("COCO", "MPPE", "HigherHRNet", 0)

    transform = cfg.dataset.TransformClass(**cfg.dataloader.transform.to_dict())

    ds = cfg.dataset.DatasetClass(
        cfg.dataset.root, "train", transform, cfg.hm_resolutions
    )

    ds.explore(idx=0, hm_idx=1)
