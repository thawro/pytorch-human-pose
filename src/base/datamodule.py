"""DataModule used to load DataLoaders"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Literal

from .transforms.transforms import ImageTransform


class DataModule:
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset | None,
        transform: ImageTransform,
        batch_size: int = 12,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn=None,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._shuffle = None

        self.dl_params = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        self.datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
        self.total_batches = {}

    def _dataloader(
        self,
        use_distributed: bool,
        split: Literal["train", "val", "test"],
    ):
        shuffle = split == "train"
        dataset = self.datasets[split]
        if use_distributed:
            params = dict(
                shuffle=False, sampler=DistributedSampler(dataset, shuffle=shuffle)
            )
        else:
            params = dict(shuffle=shuffle)
        dataloader = DataLoader(dataset, **self.dl_params, **params)
        self.total_batches[split] = len(dataloader)
        return dataloader

    def train_dataloader(self, use_distributed: bool):
        return self._dataloader(use_distributed, "train")

    def val_dataloader(self, use_distributed: bool):
        return self._dataloader(use_distributed, "val")

    def test_dataloader(self, use_distributed: bool):
        return self._dataloader(use_distributed, "test")

    def state_dict(self) -> dict:
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(state_dict["torch_cuda_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])
