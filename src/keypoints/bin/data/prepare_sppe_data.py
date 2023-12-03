from src.utils.config import DS_ROOT
import glob
from src.utils.files import load_yaml, save_yaml
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import math

RESOLUTION = (256, 256)


def process_mpii_obj_annot(annot: dict) -> tuple[tuple[int, int, int, int], bool]:
    kpts_x = []
    kpts_y = []

    for kpt in annot["keypoints"]:
        if kpt["x"] <= 0 and kpt["y"] <= 0:
            pass
        else:
            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            kpts_x.append(kpt["x"] - 1)
            kpts_y.append(kpt["y"] - 1)
    ymin, ymax = min(kpts_y), max(kpts_y)
    xmin, xmax = min(kpts_x), max(kpts_x)

    y_size = ymax - ymin
    x_size = xmax - xmin

    is_valid = y_size > 0 and x_size > 0

    return (xmin, ymin, xmax, ymax), is_valid


def process_coco_obj_annot(annot: dict) -> tuple[tuple[int, int, int, int], bool]:
    kpts_coords = [(kpt["x"], kpt["y"]) for kpt in annot["keypoints"]]
    is_crowd = annot["iscrowd"]

    valid_kpts = [kpt[0] > 0 and kpt[1] > 0 for kpt in kpts_coords]
    is_valid = not is_crowd and sum(valid_kpts) >= 4

    xmin, ymin, box_w, box_h = annot["bbox"]
    xmin, ymin = int(xmin), int(ymin)
    xmax, ymax = xmin + int(box_w), ymin + int(box_h)

    return (xmin, ymin, xmax, ymax), is_valid


def prepare_sppe_data(new_resolution: tuple[int, int], ds_name: str):
    new_h, new_w = new_resolution
    aspect_ratio = new_h / new_w
    org_dirname = "HumanPose"
    sppe_dirname = "SPPEHumanPose"
    ds_root = str(DS_ROOT / ds_name / org_dirname)
    new_ds_root = DS_ROOT / ds_name / sppe_dirname

    splits = ["train", "val"]

    for split in tqdm(splits):
        annot_filepaths = sorted(glob.glob(f"{ds_root}/annots/{split}/*"))
        split_annots_path = new_ds_root / "annots" / split
        split_imgs_path = new_ds_root / "images" / split
        split_annots_path.mkdir(parents=True, exist_ok=True)
        split_imgs_path.mkdir(parents=True, exist_ok=True)
        for annot_path in tqdm(annot_filepaths, desc=split):
            filename = Path(annot_path).stem
            img_path = annot_path.replace("annots", "images").replace(".yaml", ".jpg")
            image = np.asarray(Image.open(img_path), dtype=np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w = image.shape[:2]
            annot = load_yaml(annot_path)
            objects = annot["objects"]
            for i, obj in enumerate(objects):
                new_filename = f"{filename}_{i}"
                new_img_path = split_imgs_path / f"{new_filename}.jpg"
                new_annot_path = split_annots_path / f"{new_filename}.yaml"

                if ds_name == "MPII":
                    process_annot = process_mpii_obj_annot
                else:
                    process_annot = process_coco_obj_annot
                (xmin, ymin, xmax, ymax), is_valid = process_annot(obj)

                if not is_valid:
                    continue

                y_size = ymax - ymin
                x_size = xmax - xmin

                xc = xmin + x_size // 2
                yc = ymin + y_size // 2

                y_marg = int(0.25 * y_size)
                y_size += y_marg

                x_marg = int(0.25 * x_size)
                x_size += x_marg

                size = max(y_size, x_size)

                # x_size = int(y_size / aspect_ratio)
                x_size, y_size = size, size

                new_image = np.zeros((y_size, x_size, 3), dtype=np.uint8)

                xmin = max(xc - x_size // 2, 0)
                xmax = min(w, xc + x_size // 2)
                ymin = max(yc - y_size // 2, 0)
                ymax = min(h, yc + y_size // 2)

                obj_image = image[ymin:ymax, xmin:xmax]

                obj_w = xmax - xmin
                obj_h = ymax - ymin

                pad_x = math.floor((x_size - obj_w) / 2)
                pad_y = math.floor((y_size - obj_h) / 2)

                new_ymin, new_ymax = pad_y, pad_y + obj_image.shape[0]
                new_xmin, new_xmax = pad_x, pad_x + obj_image.shape[1]

                new_image[new_ymin:new_ymax, new_xmin:new_xmax] = obj_image
                new_image = cv2.resize(new_image, (new_w, new_h))
                Image.fromarray(new_image).save(new_img_path)

                kpt_x_offset = -xmin + pad_x
                kpt_y_offset = -ymin + pad_y
                kpt_y_scaler = new_h / y_size
                kpt_x_scaler = new_w / x_size

                kpts = obj["keypoints"]
                new_kpts = []
                for kpt in kpts:
                    if kpt["x"] <= 0 and kpt["y"] <= 0:
                        new_x = 0
                        new_y = 0
                    else:
                        new_x = max(int((kpt["x"] + kpt_x_offset) * kpt_x_scaler), 1)
                        new_y = max(int((kpt["y"] + kpt_y_offset) * kpt_y_scaler), 1)
                    vis = new_x > 0 and new_y > 0
                    if ds_name == "MPII":  # matlab 1-index notation
                        new_x -= 1
                        new_y -= 1
                    new_kpts.append({"x": new_x, "y": new_y, "visibility": vis})

                new_annot = {
                    "filename": f"{new_filename}.jpg",
                    "height": new_h,
                    "width": new_w,
                    "keypoints": new_kpts,
                }
                save_yaml(new_annot, new_annot_path)


def main():
    res = (256, 256)
    prepare_sppe_data(new_resolution=res, ds_name="COCO")


if __name__ == "__main__":
    main()
