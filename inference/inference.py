from model import ONNXSPPEInferenceModel
from process import WebcamProcessor, VideoProcessor

from utils import ROOT

VIDEO_DIR = str(ROOT / "videos")

RESULTS_PATH = "/home/thawro/Desktop/projects/pytorch-human-pose/results"
EXPERIMENT_NAME = "test"
RUN_NAME = "13-12-2023_17:26:01_SPPE_MPII_LR(0.001)_HRNet"

ONNX_PATH = f"{RESULTS_PATH}/{EXPERIMENT_NAME}/{RUN_NAME}/model/onnx/last.onnx"


def main():
    onnx_model = ONNXSPPEInferenceModel(ONNX_PATH)
    # processor = WebcamProcessor(model=onnx_model)
    video_filepath = f"{VIDEO_DIR}/sppe_0.mp4"
    processor = VideoProcessor(onnx_model, filepath=video_filepath)
    processor.process()


if __name__ == "__main__":
    main()
