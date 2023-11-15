import albumentations as A
from src.data.transforms.base import ImageTransform, _normalize
import cv2


class KeypointsImageTransform(ImageTransform):
    def __init__(
        self, mean: _normalize, std: _normalize, transform: A.Compose, size: int
    ):
        keypoint_params = A.KeypointParams(
            format="xy", label_fields=["visibilities"], remove_invisible=False
        )
        super().__init__(mean, std, transform, keypoint_params=keypoint_params)
        self.size = size


class KeypointsTransform:
    def __init__(
        self,
        mean: _normalize,
        std: _normalize,
        size: int = 512,
        multi_obj: bool = False,
    ):
        self.size = size
        self.multi_obj = multi_obj
        rotate = A.Rotate(limit=45, p=0.3)
        if multi_obj:
            smallest_max_size = A.SmallestMaxSize(size)
            train_postprocessing = A.Compose(
                [rotate, smallest_max_size, A.RandomCrop(size, size)]
            )
            inference_postprocessing = A.Compose(
                [smallest_max_size, A.CenterCrop(size, size)]
            )
        else:
            longest_max_size = A.LongestMaxSize(size)
            pad_if_needed = A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT)
            train_postprocessing = A.Compose([rotate, longest_max_size, pad_if_needed])
            inference_postprocessing = A.Compose([longest_max_size, pad_if_needed])

        self.train = KeypointsImageTransform(mean, std, train_postprocessing, size=size)
        self.inference = KeypointsImageTransform(
            mean, std, inference_postprocessing, size=size
        )
