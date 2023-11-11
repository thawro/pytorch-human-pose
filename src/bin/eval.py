"""Evaluate the model"""
import cv2
import time
import torch
import numpy as np
from functools import partial

from src.bin.config import IMGSZ, MEAN, STD, DEVICE, MODEL_INPUT_SIZE, ROOT
from src.utils.video import process_video, save_frames_to_video, get_video_size
from src.utils.image import (
    stack_frames_horizontally,
    add_txt_to_image,
    add_labels_to_frames,
    RED,
    GREEN,
    BLUE,
)
