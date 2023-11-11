"""Functions used to load model with trained weights"""
from src.model.model.base import BaseModel
from src.model.architectures.some_net import SomeNet
import torch


def load_model(input_size: tuple[int, ...], ckpt_path: str, device: str):
    model = BaseModel(
        net=SomeNet(),
        input_size=input_size,
        input_names=["images"],
        output_names=["masks", "class_probs"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    model.load_state_dict(ckpt)
    return model.to(device)
