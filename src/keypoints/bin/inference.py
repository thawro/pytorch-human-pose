import torch
from torch import nn, Tensor
from src.utils.model import seed_everything

from src.keypoints.bin.utils import create_model
from src.keypoints.bin.config import create_config
from src.utils.video import process_video
from src.keypoints.transforms import MPPEKeypointsTransform
from src.keypoints.results import InferenceMPPEKeypointsResult
from src.keypoints.visualization import plot_heatmaps, plot_connections
import cv2
import numpy as np
import albumentations as A
from src.utils.config import RESULTS_PATH
from functools import partial


class MPPEInferenceKeypointsModel(nn.Module):
    def __init__(self, net: nn.Module, device: str):
        super().__init__()
        self.net = net.to(device)
        self.device = device

        self.transform = MPPEKeypointsTransform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            symmetric_keypoints=None,
            out_size=(512, 512),
        )

    def prepare_input(self, frame: np.ndarray) -> Tensor:
        transforms = A.Compose(
            self.transform.preprocessing.transforms
            + self.transform.inference.transforms
            + self.transform.postprocessing.transforms,
        )

        transformed = transforms(image=frame)
        return transformed["image"]

    def __call__(self, frame: np.ndarray) -> InferenceMPPEKeypointsResult:
        x = self.prepare_input(frame).unsqueeze(0).to(self.device)
        image = self.transform.inverse_preprocessing(x.detach().cpu().numpy()[0])
        stages_pred_kpts_heatmaps, stages_pred_tags_heatmaps = self.net(x)
        return InferenceMPPEKeypointsResult.from_preds(
            image,
            stages_pred_kpts_heatmaps,
            stages_pred_tags_heatmaps,
            max_num_people=10,
            det_thr=0.2,
            tag_thr=1.0,
        )


def processing_fn(model: MPPEInferenceKeypointsModel, frame: np.ndarray) -> dict:
    result = model(frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("Model input", result.image)
    kpts_image = plot_connections(
        result.image, result.pred_keypoints, result.pred_scores, limbs=None, thr=0.2
    )
    cv2.imshow("Keypoints", kpts_image)
    return {}


def main() -> None:
    cfg = create_config("MPII", "MPPE", "HigherHRNet")
    cfg.setup.ckpt_path = str(
        RESULTS_PATH
        / "test/01-07_21:53__sigmoid_MPPE_MPII_HigherHRNet/01-08_09:30/checkpoints/best.pt"
    )
    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    net = create_model(cfg).net
    model = MPPEInferenceKeypointsModel(net, device="cuda:0")
    ckpt = torch.load(cfg.setup.ckpt_path)
    model.load_state_dict(ckpt["module"]["model"])
    process_video(partial(processing_fn, model=model), filename=0)


if __name__ == "__main__":
    main()
