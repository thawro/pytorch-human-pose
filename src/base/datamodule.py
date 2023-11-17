"""DataModule used to load DataLoaders"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset


from .transforms import BaseTransform


class DataModule:
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset | None,
        transform: BaseTransform,
        batch_size: int = 12,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
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

        dl_params = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            batch_size=batch_size,
        )

        self.train_dataloader = DataLoader(train_ds, shuffle=True, **dl_params)
        self.val_dataloader = DataLoader(val_ds, shuffle=False, **dl_params)
        self.total_batches = {
            "train": len(self.train_dataloader),
            "val": len(self.val_dataloader),
        }

        if test_ds is not None:
            self.test_dataloader = DataLoader(test_ds, shuffle=False, **dl_params)
            self.total_batches["test"] = len(self.test_dataloader)

    def state_dict(self) -> dict:
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])
