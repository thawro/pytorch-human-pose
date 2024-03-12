"""DataModule used to load DataLoaders"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.logger.pylogger import log
from src.utils.image import make_grid
from src.utils.types import _split


class DataModule:
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset | None,
        batch_size: int = 12,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn=None,
        use_DDP: bool = False,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._shuffle = None
        self.use_DDP = use_DDP

        self.dl_params = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        self.datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
        self._log_statistics()
        self.total_batches = {}
        self.train_dataloader = self._dataloader("train")
        self.val_dataloader = self._dataloader("val")
        if test_ds is not None:
            self.test_dataloader = self._dataloader("test")

    def _log_statistics(self):
        statistics_repr = []
        for split, ds in self.datasets.items():
            if ds is not None:
                ds_repr = f"{split}: {len(ds)} samples ({ds.__class__.__name__})"

            else:
                ds_repr = f"{split}: 0 (dataset is None)"
            statistics_repr.append(ds_repr)
        statistics_repr = "\n".join(statistics_repr)
        log.info(f"DataModule statistics:\n{statistics_repr}")

    def _dataloader(self, split: _split):
        shuffle = split == "train"
        dataset = self.datasets[split]

        if self.use_DDP:
            log.info("..Using DistributedSampler..")
            rank = int(os.environ["LOCAL_RANK"])
            params = dict(
                shuffle=False,
                sampler=DistributedSampler(
                    dataset,
                    shuffle=shuffle,
                    rank=rank,
                    drop_last=self.dl_params.get("drop_last", False),
                ),
            )
        else:
            log.info("..Using default Sampler..")
            params = dict(shuffle=shuffle)
        dataloader = DataLoader(dataset, **self.dl_params, **params)
        self.total_batches[split] = len(dataloader)
        return dataloader

    def state_dict(self) -> dict:
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "torch_cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "torch_cuda_random_state": torch.cuda.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(state_dict["torch_cuda_random_state_all"])
        # torch.cuda.set_rng_state(state_dict["torch_cuda_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])
        log.info("     Loaded datamodule state")

    def explore(self):
        import cv2

        dataloader = iter(self.train_dataloader(use_distributed=False))

        def inf_dl_gen():
            while True:
                for batch in dataloader:
                    images, annots, class_idxs = batch
                    images = [self.transform.inverse_preprocessing(img) for img in images]
                    images = make_grid(images, nrows=3)
                    yield images

        gen = inf_dl_gen()
        images = next(gen)
        while True:
            cv2.imshow("Sample", cv2.cvtColor(images, cv2.COLOR_RGB2BGR))
            k = cv2.waitKeyEx(0)
            right_key = 65363
            if k % 256 == 27:  # ESC pressed
                print("Escape hit, closing")
                cv2.destroyAllWindows()
                break
            elif k % 256 == 32 or k == right_key:  # SPACE or right arrow pressed
                print("Space or right arrow hit, exploring next sample")
                images = next(gen)
