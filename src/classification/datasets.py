from src.base.datasets import BaseImageDataset
from src.classification.transforms import ClassificationTransform
from src.utils.image import make_grid, put_txt, YELLOW
import numpy as np
import cv2
import glob
import yaml
import xml.dom.minidom


class ClassificationDataset(BaseImageDataset):
    transform: ClassificationTransform
    labels: list[str]

    def __init__(
        self, root: str, split: str, transform: ClassificationTransform | None
    ):
        super().__init__(root, split, transform)
        self.out_size = transform.out_size

    def _transform(self, image: np.ndarray) -> np.ndarray:
        if self.is_train:
            transform = self.transform.random
        else:
            transform = self.transform.inference
        transformed = transform(image=image)
        transformed = self.transform.preprocessing(**transformed)
        transformed = self.transform.postprocessing(**transformed)
        return transformed["image"]

    def plot(self, idx: int):
        raw_image, raw_annot = self.get_raw_data(idx)
        raw_h, raw_w = raw_image.shape[:2]

        image, annot, class_idx = self[idx]
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

        _txt_params = dict(loc="br", txt_color=YELLOW, vspace=2)
        put_txt(raw_image, ["Raw", f"{raw_h}x{raw_w}"], **_txt_params)
        put_txt(image, ["Transformed", f"{tr_h}x{tr_w}"], **_txt_params)

        grid = make_grid([raw_image, image], nrows=1, resize=1.5)
        f_xy = 1.2
        grid = cv2.resize(grid, dsize=(0, 0), fx=f_xy, fy=f_xy)
        labels = [
            annot["filename"],
            f"label: {annot['label']}",
            f"class_idx: {annot['class_idx']}",
        ]
        put_txt(grid, labels, loc="tc")
        return grid

    def get_raw_data(self, idx) -> tuple[np.ndarray, dict]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        return image, annot

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict, int]:
        image, annot = self.get_raw_data(idx)
        image = image[..., :3]  # rgba -> rgb
        image = self._transform(image)
        label_info = annot.pop("label")
        annot.update(
            {
                "label": label_info["label"],
                "class_idx": label_info["class_idx"],
                "wordnet_name": label_info["wordnet_name"],
            }
        )
        class_idx = annot["class_idx"]
        return image, annot, class_idx


class ImageNetClassificationDataset(ClassificationDataset):
    """
    Class idxs from: https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
    """

    def __init__(
        self, root: str, split: str, transform: ClassificationTransform | None
    ):
        super().__init__(root, split, transform)
        self.wordnet2label = self._create_wordnet_labels()
        self.labels = [
            label_info["label"] for label_info in self.wordnet2label.values()
        ]

    def _create_wordnet_labels(self) -> dict[str, dict[str, int | str]]:
        with open(f"{self.root}/class_index.yaml", "r") as f:
            class_index = yaml.safe_load(f)
        wordnet2label = {}
        for idx, (wordnet_label, label) in class_index.items():
            label_info = {"class_idx": int(idx), "label": label}
            wordnet2label[wordnet_label] = label_info
        return wordnet2label

    def get_images_annots_filepaths(self) -> tuple[list[str], list[str]]:
        annots_dir = f"{str(self.root)}/ILSVRC/Annotations/CLS-LOC/{self.split}"
        if self.is_train:
            annots_filepaths = sorted(glob.glob(f"{annots_dir}/*/*"))
        else:
            annots_filepaths = sorted(glob.glob(f"{annots_dir}/*"))
        images_filepaths = [
            path.replace("Annotations/", "Data/").replace(".xml", ".JPEG")
            for path in annots_filepaths
        ]
        return images_filepaths, annots_filepaths

    def load_annot(self, idx: int) -> dict:
        """
        Returned label is in form:
        {
            'label': {'wordnet_name': str, 'label': str, 'class_idx': int},
            'truncated': bool,
            'difficult': bool,
            "filename": str,
            'bbox_xyxy': list[int], # [xmin, ymin, xmax, ymax]
            'height': int,
            'width': int
        }
        """
        annot_path = self.annots_filepaths[idx]
        root = xml.dom.minidom.parse(annot_path).documentElement
        obj = root.getElementsByTagName("object")[0]
        annot = {
            key: obj.getElementsByTagName(key)[0].childNodes[0].data
            for key in ["name", "truncated", "difficult"]
        }
        bbox = obj.getElementsByTagName("bndbox")[0]
        bbox_xyxy = [
            int(bbox.getElementsByTagName(key)[0].childNodes[0].data)
            for key in ["xmin", "ymin", "xmax", "ymax"]
        ]
        annot["bbox_xyxy"] = bbox_xyxy

        size = root.getElementsByTagName("size")[0]
        h, w = [
            int(size.getElementsByTagName(key)[0].childNodes[0].data)
            for key in ["height", "width"]
        ]
        filename = root.getElementsByTagName("filename")[0].childNodes[0].data
        annot.update({"height": h, "width": w, "filename": filename})
        wordnet_name = annot.pop("name")
        label_info = self.wordnet2label[wordnet_name]
        annot["label"] = {
            "wordnet_name": wordnet_name,
            "label": label_info["label"],
            "class_idx": label_info["class_idx"],
        }
        for key in ["truncated", "difficult"]:
            annot[key] = bool(int(annot[key]))
        return annot


def test_dataset(
    dataset: BaseImageDataset, batch_size: int = 96, num_workers: int = 16
):
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    for idx, batch in tqdm(enumerate(dl), total=len(ds) // batch_size):
        pass


if __name__ == "__main__":
    from src.utils.config import DS_ROOT

    ds = ImageNetClassificationDataset(
        root=str(DS_ROOT / "ImageNet"),
        split="train",
        transform=ClassificationTransform(),
    )

    ds.explore(idx=0)
