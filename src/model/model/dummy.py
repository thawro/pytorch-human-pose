from torch import Tensor, nn
from .base import BaseModel
import torch


class DummyModel(BaseModel):
    net: nn.Module

    def __init__(self):
        net = nn.Linear(1, 1)
        input_names = ["input"]
        output_names = ["output"]
        super().__init__(net, input_names, output_names)

    def example_input(self, batch_size: int = 1) -> dict[str, Tensor]:
        return {"x": torch.randn(batch_size, 1, device=self.device)}

    def predict(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return out
