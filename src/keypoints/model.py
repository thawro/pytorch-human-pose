import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn

from src.base.model import BaseImageModel, BaseInferenceModel
from src.base.transforms.utils import resize_align_multi_scale
from src.logger.pylogger import log

from .datasets import COCO_LIMBS
from .results import InferenceKeypointsResult
from .transforms import COCO_FLIP_INDEX


class KeypointsModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["keypoints"])

    def init_weights(self):
        log.info("..Initializing weights from normal distribution (Keypoints)..")
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 512, 512)


class InferenceKeypointsModel(BaseInferenceModel):
    limbs = COCO_LIMBS

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        net: nn.Module,
        det_thr: float = 0.05,
        tag_thr: float = 0.5,
        use_flip: bool = False,
        input_size: int = 512,
        max_num_people: int = 30,
        device: str = "cuda:0",
        ckpt_path: str | None = None,
    ):
        super().__init__(net, input_size, device, ckpt_path)
        self.det_thr = det_thr
        self.tag_thr = tag_thr
        self.max_num_people = max_num_people
        self.use_flip = use_flip

    def prepare_input(
        self, image: np.ndarray
    ) -> tuple[Tensor, tuple[int, int], tuple[float, float]]:
        image_resized, center, scale = resize_align_multi_scale(image, self.input_size, 1, 1)
        image_resized = self.transform(image_resized)
        x = image_resized.unsqueeze(0).to(self.device)
        return x, center, scale

    def __call__(self, raw_image: np.ndarray, annot: list[dict] | None) -> InferenceKeypointsResult:
        with torch.no_grad():
            x, center, scale = self.prepare_input(raw_image)
            kpts_heatmaps, tags_heatmaps = self.net(x)

            if self.use_flip:
                flip_kpts_heatmaps, flip_tags_heatmaps = self.net(torch.flip(x, [3]))
                for i in range(len(kpts_heatmaps)):
                    pred_hms = kpts_heatmaps[i]
                    flip_pred_hms = torch.flip(flip_kpts_heatmaps[i], [3])
                    kpts_heatmaps[i] = (pred_hms + flip_pred_hms[:, COCO_FLIP_INDEX]) / 2
                tags_heatmaps = [
                    tags_heatmaps,
                    torch.flip(flip_tags_heatmaps, [3])[:, COCO_FLIP_INDEX],
                ]
            else:
                tags_heatmaps = [tags_heatmaps]

        model_input_image = x[0]
        return InferenceKeypointsResult.from_preds(
            raw_image=raw_image,
            annot=annot,
            model_input_image=model_input_image,
            kpts_heatmaps=kpts_heatmaps,
            tags_heatmaps=tags_heatmaps,
            limbs=self.limbs,
            scale=scale,
            center=center,
            det_thr=self.det_thr,
            tag_thr=self.tag_thr,
            max_num_people=self.max_num_people,
        )
