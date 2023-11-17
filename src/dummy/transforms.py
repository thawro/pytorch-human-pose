import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class DummyTransform:
    def __init__(self):
        self.preprocessing = A.Compose([])
        self.random = A.Compose([])
        self.postprocessing = A.Compose([ToTensorV2(transpose_mask=True)])

    @property
    def inverse_preprocessing(self):
        def transform(x: np.ndarray | Image.Image):
            """Apply inverse of preprocessing to the image (for visualization purposes)."""
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(x, Image.Image):
                x = np.array(x)
            return x

        return transform
