import torch
from torch import nn, Tensor
import torchvision.transforms as T
from src.utils.model import seed_everything

from src.classification.config import ClassificationConfig
import cv2
import numpy as np
from src.utils.config import RESULTS_PATH, DS_ROOT, YAML_EXP_PATH
from functools import partial
from src.classification.datasets import ImageNetClassificationDataset

from src.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(int(224 / 0.875)),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InferenceClassificationModel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        transform: T.Compose,
        labels: list[str],
        device: str = "cuda:1",
    ):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.labels = np.array(labels)
        self.transform = transform

    def prepare_input(self, image: np.ndarray) -> Tensor:
        image_resized = self.transform(image)
        x = image_resized.unsqueeze(0).to(self.device)
        return x

    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = self.prepare_input(image)
        with torch.no_grad():
            logits = self.net(x).squeeze()

        top_5 = torch.topk(logits, k=5).indices.cpu().numpy()

        input_image = x[0].permute(1, 2, 0).cpu().numpy()
        _mean = np.array([0.485, 0.456, 0.406]) * 255
        _std = np.array([0.229, 0.224, 0.225]) * 255
        input_image = (input_image * _std) + _mean
        input_image = input_image.astype(np.uint8)

        print(self.labels[top_5])
        return input_image, top_5


def processing_fn(
    model: InferenceClassificationModel, frame: np.ndarray, annot
) -> dict:
    print(annot)
    with torch.no_grad():
        input_image, logits = model(frame)

    print("=" * 100)
    cv2.imshow(
        "Output",
        cv2.cvtColor(
            cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR
        ),
    )
    print(logits)
    return {}


def load_model(
    cfg: ClassificationConfig, ckpt_path: str, labels: list[str]
) -> InferenceClassificationModel:
    cfg.setup.is_train = False
    cfg.setup.ckpt_path = ckpt_path
    device_id = 0
    device = f"cuda:{device_id}"

    net = cfg.create_net()
    model = InferenceClassificationModel(net, transform, labels=labels, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    for key in list(ckpt.keys()):
        ckpt[key.replace("module.1.", "").replace("module.", "")] = ckpt[key]
        ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
    log.info(f"Loaded model from {ckpt_path}")
    return model


def main() -> None:
    seed_everything(42)
    ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-09_23:08___ImageNet_HRNet/02-10_07:47/checkpoints/best.pt"

    cfg_path = str(YAML_EXP_PATH / "classification" / "hrnet_32.yaml")
    cfg = ClassificationConfig.from_yaml(cfg_path)
    datamodule = cfg.create_datamodule()
    ds = datamodule.val_ds

    model = load_model(cfg, ckpt_path, labels=ds.labels)

    ds.perform_inference(partial(processing_fn, model=model))


if __name__ == "__main__":
    main()
