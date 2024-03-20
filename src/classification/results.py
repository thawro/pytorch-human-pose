from dataclasses import dataclass

import numpy as np
from torch import Tensor

from src.base.results import BaseResult
from src.classification.visualization import plot_top_preds
from src.utils.image import resize_with_aspect_ratio

from .transforms import ClassificationTransform


class ClassificationResult(BaseResult):
    def __init__(
        self,
        model_input_image: Tensor,
        logits: Tensor,
        target_label: int | str | None,
        idx2label: dict[int, str],
    ):
        self.model_input_image = ClassificationTransform.inverse_transform(model_input_image)
        self.target_label = target_label
        self.idx2label = idx2label
        self.logits = logits.numpy()

    def plot(self) -> dict[str, np.ndarray]:
        exp_logits = np.exp(self.logits) + np.finfo(self.logits.dtype).eps
        probs = exp_logits / np.sum(exp_logits)
        top_preds_plot = plot_top_preds(
            image=self.model_input_image,
            probs=probs,
            idx2label=self.idx2label,
            k=5,
            target_label=self.target_label,
        )
        return {"top_probs": top_preds_plot}


@dataclass
class InferenceClassificationResult(BaseResult):
    raw_image: np.ndarray
    model_input_image: np.ndarray
    logits: np.ndarray
    target_label: int | str | None
    pred_label: int | str
    idx2label: dict[int, str]

    @classmethod
    def from_preds(
        cls,
        raw_image: np.ndarray,
        model_input_image: Tensor,
        logits: Tensor,
        idx2label: dict[int, str],
        target_label: int | str | None = None,
    ):
        pred_idx = int(logits.argmax().item())
        pred_label = idx2label[pred_idx]

        model_input_image_npy = ClassificationTransform.inverse_transform(model_input_image)
        return cls(
            raw_image=raw_image,
            model_input_image=model_input_image_npy,
            logits=logits.cpu().numpy(),
            target_label=target_label,
            pred_label=pred_label,
            idx2label=idx2label,
        )

    def plot(self) -> dict[str, np.ndarray]:
        exp_logits = np.exp(self.logits)
        probs = exp_logits / np.sum(exp_logits)
        raw_image = self.raw_image.copy()
        raw_image = resize_with_aspect_ratio(raw_image, height=512, width=None)
        top_preds_plot = plot_top_preds(
            image=raw_image,
            probs=probs,
            idx2label=self.idx2label,
            k=5,
            target_label=self.target_label,
        )
        return {"top_preds": top_preds_plot}
