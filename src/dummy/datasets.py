import torch
from torch import Tensor

from src.base.datasets import BaseDataset


class DummyDataset(BaseDataset):
    def __init__(self, root: str, split: str = "test", transform=None):
        super().__init__(root, split, transform=transform)
        self.data = torch.randn(1000, 1)
        self.targets = self.data * 2 + 1  # f(x) = 2x + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        data, target = self.data[idx].clone(), self.targets[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data, target
