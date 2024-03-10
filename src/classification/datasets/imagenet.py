import glob

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image

from src.base.datasets import ExplorerDataset, InferenceDataset
from src.classification.transforms import ClassificationTransform
from src.utils.files import load_yaml
from src.utils.image import put_txt


def parse_wordnet_labels(
    wordnet_labels: dict[str, list[str]],
) -> tuple[dict[int, str], dict[str, int]]:
    wordnet2idx = {}
    idx2label = {}
    for idx, labels in wordnet_labels.items():
        # labels is a list [<wordnet_label>, <label>]
        # <wordnet_label> <idx> <label_name>
        wordnet, label = labels
        idx = int(idx)
        wordnet2idx[wordnet] = idx
        idx2label[idx] = label
    return idx2label, wordnet2idx


class ImagenetClassificationDataset(datasets.ImageFolder, ExplorerDataset, InferenceDataset):
    # labels from:
    # https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
    idx2label: dict[int, str]
    wordnet2idx: dict[str, int]

    def __init__(self, root: str, split: str, transform: T.Compose | None = None):
        ds_root = f"{root}/{split}"
        datasets.ImageFolder.__init__(self, ds_root, transform=transform)
        # self.idx2label = load_yaml(f"{root}/idx2label.yaml")
        wordnet_labels = load_yaml(f"{root}/wordnet_labels.yaml")
        self.idx2label, self.wordnet2idx = parse_wordnet_labels(wordnet_labels)
        self.split = split
        self._set_paths()

    def _set_paths(self):
        images_filepaths = glob.glob(f"{self.root}/*/*")
        self.images_filepaths = np.array(images_filepaths, dtype=np.str_)

    def load_image(self, idx: int) -> np.ndarray:
        return np.array(Image.open(self.images_filepaths[idx]).convert("RGB"))

    def load_annot(self, idx: int, use_wordnet: bool = False) -> str:
        image_filepath = self.images_filepaths[idx]
        wordnet_label = image_filepath.split("/")[-2]
        if use_wordnet:
            return wordnet_label
        idx = self.wordnet2idx[wordnet_label]
        return self.idx2label[idx]

    def plot(self, idx: int, **kwargs) -> np.ndarray:
        image, label = self[idx]
        image = ClassificationTransform.inverse_transform(image)
        wordnet_label = self.classes[label]
        label = self.idx2label[self.wordnet2idx[wordnet_label]]
        image = put_txt(image.copy(), [f"Wordnet label: {wordnet_label}", f"Label: {label}"])
        return image


if __name__ == "__main__":
    from PIL import Image

    from src.classification.transforms import ClassificationTransform

    transform = ClassificationTransform()
    ds = ImagenetClassificationDataset("data/ImageNet", "train", transform.train)
    grid = ds.plot_examples([0, 1, 2, 3, 4, 5, 6], nrows=1, stage_idxs=[1, 0])
    Image.fromarray(grid).save("test.jpg")
