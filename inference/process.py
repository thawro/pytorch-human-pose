import cv2
import numpy as np
from model import ONNXInferenceModel
from abc import abstractmethod
from visualization import visualize
from geda.data_providers.mpii import LIMBS


class InferenceProcessor:
    def __init__(self, model: ONNXInferenceModel):
        self.model = model

    @abstractmethod
    def next(self) -> tuple[bool, np.ndarray]:
        raise NotImplementedError()

    def process(self):
        while True:
            success, frame = self.next()
            if not success:
                break
            frame_input = self.model.transform_images([frame])
            frame_rgb = cv2.cvtColor(
                self.model.inverse_transform(frame_input[0]), cv2.COLOR_RGB2BGR
            )
            heatmaps, joints = self.model(frame_input)
            all_kpts_coords = joints[..., :2].astype(np.int32)
            all_kpts_scores = joints[..., 2]
            grid = visualize(
                frame_rgb,
                heatmaps,
                all_kpts_coords,
                all_kpts_scores,
                limbs=LIMBS,
                thr=0.1,
            )
            cv2.imshow("Keypoints", grid)
        self.onend()

    def onend(self):
        pass


class OpenCVProcessor(InferenceProcessor):
    def __init__(self, model: ONNXInferenceModel, cap_input: int | str):
        super().__init__(model)
        self.cap = cv2.VideoCapture(cap_input)

    def next(self) -> tuple[bool, np.ndarray]:
        success, frame = self.cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            success = False

        return success, frame_rgb

    def onend(self):
        self.cap.release()
        cv2.destroyAllWindows()


class WebcamProcessor(OpenCVProcessor):
    def __init__(self, model: ONNXInferenceModel):
        super().__init__(model, cap_input=0)


class VideoProcessor(OpenCVProcessor):
    def __init__(self, model: ONNXInferenceModel, filepath: str):
        super().__init__(model, cap_input=filepath)
