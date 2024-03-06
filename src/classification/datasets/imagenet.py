import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as T

from src.base.datasets import ExplorerDataset
from src.utils.image import put_txt


class ImagenetClassificationDataset(datasets.ImageFolder, ExplorerDataset):
    def __init__(self, root: str, split: str, transform: T.Compose):
        ds_root = f"{root}/{split}"
        super().__init__(ds_root, transform)

    def plot(self, idx: int, **kwargs) -> np.ndarray:
        img, label = self[idx]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img.permute(1, 2, 0).numpy() * std) + mean
        img = (img * 255).astype(np.uint8)
        txt_label = f"Label: {self.classes[label]}"
        img = put_txt(img.copy(), [txt_label])
        return img


if __name__ == "__main__":
    from PIL import Image

    from src.classification.transforms import ClassificationTransform

    transform = ClassificationTransform()
    # ds = CocoKeypoints("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = ImagenetClassificationDataset("data/ImageNet", "train", transform.train)
    grid = ds.plot_examples([0, 1, 2, 3, 4, 5, 6], nrows=1, stage_idxs=[1, 0])
    Image.fromarray(grid).save("test.jpg")
