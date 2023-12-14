import onnxruntime as ort
import numpy as np
import cv2
from abc import abstractmethod
import torch
import torchvision.transforms.functional as F
from grouping import SPPEHeatmapParser


class ONNXInferenceModel:
    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider"]
        self.model = ort.InferenceSession(model_path, providers=providers)

    @abstractmethod
    def transform_images(self, images: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def inverse_transform(self, image: np.ndarray) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        unnorm_image = image.transpose(1, 2, 0) * std + mean

        return unnorm_image.astype(np.uint8)

    def __call__(self, images: np.ndarray):
        return self.model.run(None, {"images": images})[0]


class ONNXSPPEInferenceModel(ONNXInferenceModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.hm_parser = SPPEHeatmapParser(num_kpts=16, det_thr=0.2)

    def transform_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Prepare model input

        Transform list of HWC uint8 image frames to single NCHW float array
        """
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255

        def transform_image(image: np.ndarray) -> np.ndarray:
            dst_size = 256
            h, w = image.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > 1:  # wider
                new_w = int(dst_size * aspect_ratio)
                new_h = dst_size
            else:  # taller
                new_w = dst_size
                new_h = int(dst_size / aspect_ratio)
            resized_image = cv2.resize(image, (new_w, new_h))

            if aspect_ratio > 1:
                crop_x = (new_w - dst_size) // 2
                cropped_image = resized_image[:, crop_x : crop_x + dst_size]
            else:
                crop_y = (new_h - dst_size) // 2
                cropped_image = resized_image[crop_y : crop_y + dst_size]

            norm_image = (cropped_image - mean) / std
            return norm_image.transpose(2, 0, 1).astype(np.float32)

        transformed_images = list(map(transform_image, images))
        return np.stack(transformed_images)

    def __call__(self, images: np.ndarray):
        heatmaps = super().__call__(images)
        h, w = images.shape[-2:]
        heatmaps = torch.from_numpy(heatmaps)
        heatmaps = F.resize(heatmaps, size=[h, w])
        joints = self.hm_parser.parse(heatmaps)
        return heatmaps.numpy()[0], joints
