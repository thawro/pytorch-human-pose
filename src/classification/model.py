import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn

from src.base.model import BaseImageModel, BaseInferenceModel
from src.logger.pylogger import log

from .results import InferenceClassificationResult


class BaseClassificationModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["logits"])

    def init_weights(self):
        log.info("..Initializing weights from normal distribution (Classification)..")
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ClassificationModel(BaseClassificationModel):
    def forward(self, images: Tensor) -> list[Tensor]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 224, 224)


class InferenceClassificationModel(BaseInferenceModel):
    def __init__(
        self,
        net: nn.Module,
        idx2label: dict[int, str],
        input_size: int = 256,
        device: str = "cuda:0",
        ckpt_path: str | None = None,
    ):
        super().__init__(net, input_size, device, ckpt_path)
        self.idx2label = idx2label
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_input(self, image: np.ndarray) -> Tensor:
        x = self.transform(image)
        x = x.unsqueeze(0).to(self.device)
        return x

    def __call__(
        self, raw_image: np.ndarray, target_label: int | str | None = None
    ) -> InferenceClassificationResult:
        x = self.prepare_input(raw_image)
        with torch.no_grad():
            logits = self.net(x)

        return InferenceClassificationResult.from_preds(
            raw_image=raw_image,
            model_input_image=x[0],
            logits=logits[0],
            target_label=target_label,
            idx2label=self.idx2label,
        )
