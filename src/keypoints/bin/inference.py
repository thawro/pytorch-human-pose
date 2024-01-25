import torch
from torch import nn, Tensor
from src.utils.model import seed_everything

from src.keypoints.bin.utils import create_model
from src.keypoints.bin.config import create_config
from src.keypoints.results import InferenceMPPEKeypointsResult
from src.keypoints.datasets import coco_symmetric_labels
import cv2
import numpy as np
from src.utils.config import RESULTS_PATH, DS_ROOT
from functools import partial
from src.base.datasets import BaseImageDataset
from src.keypoints.datasets import coco_limbs, mpii_limbs
import torch

from src.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


def transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_multi_scale_size(image, input_size, current_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63) // 64 * 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size / w * h + 63) // 64 * 64) * current_scale / min_scale
        )
        scale_w = w
        scale_h = h_resized / w_resized * w
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size / h * w + 63) // 64 * 64) * current_scale / min_scale
        )
        scale_h = h
        scale_w = w_resized / h_resized * h

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale
    )
    trans = get_affine_transform(center, scale, 0, size_resized)
    image_resized = cv2.warpAffine(image, trans, size_resized)
    return image_resized, center, scale


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    for person in grouped_joints[0]:
        joints = np.zeros((person.shape[0], 3))
        joints = transform_preds(person, center, scale, heatmap_size)
        final_results.append(joints)
    return final_results


class MPPEInferenceKeypointsModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.1,
        tag_thr: float = 1.0,
        device: str = "cuda:1",
        limbs: list[tuple[int, int]] = coco_limbs,
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.input_size = 512
        self.limbs = limbs

    def prepare_input(self, image: np.ndarray) -> Tensor:
        import torchvision

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        base_size, center, scale = get_multi_scale_size(image, self.input_size, 1.0, 1)

        image_resized, center, scale = resize_align_multi_scale(
            image, self.input_size, 1, 1
        )

        image_resized = transforms(image_resized)
        x = image_resized.unsqueeze(0).to(self.device)
        return x, center, scale

    def __call__(self, image: np.ndarray, annot) -> InferenceMPPEKeypointsResult:
        x, center, scale = self.prepare_input(image)

        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.net(x)

        input_image = x[0].permute(1, 2, 0).cpu().numpy()
        _mean = np.array([0.485, 0.456, 0.406]) * 255
        _std = np.array([0.229, 0.224, 0.225]) * 255
        input_image = (input_image * _std) + _mean
        input_image = input_image.astype(np.uint8)

        return InferenceMPPEKeypointsResult.from_preds(
            annot,
            input_image,
            image,
            scale,
            center,
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            get_final_preds,
            self.limbs,
            max_num_people=30,
            det_thr=self.det_thr,
            tag_thr=self.tag_thr,
        )


def processing_fn(
    model: MPPEInferenceKeypointsModel,
    frame: np.ndarray,
    annot,
) -> dict:
    with torch.no_grad():
        result = model(frame, annot)

    print("=" * 100)
    final_plot, raw_image = result.plot()
    cv2.imshow(
        "grid",
        cv2.cvtColor(cv2.resize(final_plot, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR),
    )
    cv2.imshow("Pred", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
    return {}


def load_model(dataset: str = "COCO"):
    device_id = 1
    device = f"cuda:{device_id}"
    limbs = coco_limbs if dataset == "COCO" else mpii_limbs
    if dataset == "COCO":
        ckpt_path = str(
            RESULTS_PATH
            / "test/01-12_15:17__sigmoid_MPPE_COCO_HigherHRNet/01-14_20:44/checkpoints/last.pt"
        )
        ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/01-17_16:04__sigmoid_MPPE_COCO_HigherHRNet/01-18_11:10/checkpoints/best.pt"
        ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/01-21_11:03__org_mosaic_MPPE_COCO_OriginalHigherHRNet/01-23_08:03/checkpoints/best.pt"
        # ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/test/01-23_17:59___MPPE_COCO_OriginalHigherHRNet/01-25_08:32/checkpoints/best.pt"
    else:
        ckpt_path = str(
            RESULTS_PATH
            / "test/01-10_13:21__sigmoid_MPPE_MPII_HigherHRNet/01-11_09:10/checkpoints/last.pt"
        )
    model = "OriginalHigherHRNet"
    cfg = create_config(
        dataset,
        "MPPE",
        model,
        device_id,
        ckpt_path=ckpt_path,
        distributed=False,
        is_train=False,
    )

    seed_everything(cfg.setup.seed)

    net = create_model(cfg)

    model = MPPEInferenceKeypointsModel(
        net, device=device, limbs=limbs, det_thr=0.1, tag_thr=1.0
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    for key in list(ckpt.keys()):
        ckpt[key.replace("module.1.", "")] = ckpt[key]
        ckpt.pop(key)
    model.load_state_dict(ckpt)
    log.info(f"Loaded model from {ckpt_path}")

    model.eval()
    return model


def main() -> None:
    dataset = "COCO"
    model = load_model(dataset)
    ds = BaseImageDataset(
        root=str(DS_ROOT / f"{dataset}/HumanPose"),
        split="val",
        transform=None,
    )

    ds.perform_inference(partial(processing_fn, model=model))
    # process_video(partial(processing_fn, model=model), filename=0)


if __name__ == "__main__":
    main()
