import dataclasses
from dataclasses import dataclass
from src.utils import NOW, RESULTS_PATH
from pathlib import Path
from typing import Literal, Type
from src.utils.config import DS_ROOT

from .datasets import (
    SppeMpiiDataset,
    MppeMpiiDataset,
    SppeCocoDataset,
    MppeCocoDataset,
    BaseKeypointsDataset,
)
from .transforms import SPPEKeypointsTransform, MPPEKeypointsTransform
from .results import (
    SPPEKeypointsResults,
    MPPEKeypointsResults,
    SppeMpiiKeypointsResults,
    MppeMpiiKeypointsResults,
    SppeCocoKeypointsResults,
    MppeCocoKeypointsResults,
)
from .module import SPPEKeypointsModule, MPPEKeypointsModule
from .datasets import mpii_symmetric_labels, coco_symmetric_labels

_dataset_name = Literal["MPII", "COCO"]
_mode = Literal["SPPE", "MPPE"]
_architectures = Literal["Hourglass", "SimpleBaseline", "HRNet", "HigherHRNet"]


@dataclass
class BaseConfig:
    def to_dict(self):
        fields = dataclasses.fields(self)
        dct = {}
        for field in fields:
            field_name = field.name
            field_value = getattr(self, field_name)
            if hasattr(field_value, "to_dict"):
                dct[field_name] = field_value.to_dict()
            else:
                dct[field_name] = field_value
        return dct


@dataclass
class TransformConfig(BaseConfig):
    mean: tuple[float, ...] | list[float]
    std: tuple[float, ...] | list[float]
    out_size: list[int] | tuple[int, int]
    symmetric_keypoints: list[int]


@dataclass
class DatasetConfig(BaseConfig):
    name: _dataset_name
    mode: _mode

    @property
    def DatasetClass(self) -> Type[BaseKeypointsDataset]:
        Datasets = {
            "SPPE": {"COCO": SppeCocoDataset, "MPII": SppeMpiiDataset},
            "MPPE": {"COCO": MppeCocoDataset, "MPII": MppeMpiiDataset},
        }
        return Datasets[self.mode][self.name]

    @property
    def TransformClass(self) -> Type[SPPEKeypointsTransform | MPPEKeypointsTransform]:
        if self.mode == "SPPE":
            return SPPEKeypointsTransform
        elif self.mode == "MPPE":
            return MPPEKeypointsTransform
        else:
            raise ValueError("Wrong mode passed. Possible: ['SPPE', 'MPPE']")

    @property
    def subdir(self) -> str:
        if self.mode == "SPPE":
            return "SPPEHumanPose"
        elif self.mode == "MPPE":
            return "HumanPose"
        else:
            raise ValueError("Wrong mode passed. Possible: ['SPPE', 'MPPE']")

    @property
    def root(self) -> str:
        return str(DS_ROOT / self.name / self.subdir)

    @property
    def out_size(self) -> list[int]:
        if self.mode == "SPPE":
            return [256, 256]
        else:
            return [512, 512]

    @property
    def symmetric_keypoints(self) -> list[int]:
        if self.name == "MPII":
            return mpii_symmetric_labels
        elif self.name == "COCO":
            return coco_symmetric_labels
        else:
            raise ValueError("Wrong dataset name passed. Possible: ['MPII', 'COCO']")

    def to_dict(self) -> dict:
        dct = super().to_dict()
        dct["out_size"] = self.out_size
        dct["root"] = self.root
        return dct


@dataclass
class DataloaderConfig(BaseConfig):
    batch_size: int
    transform: TransformConfig


@dataclass
class TrainerConfig(BaseConfig):
    device_id: int
    max_epochs: int
    limit_batches: int
    log_every_n_steps: int


@dataclass
class SetupConfig(BaseConfig):
    experiment_name: str
    name_prefix: str
    seed: int
    dataset: _dataset_name
    ckpt_path: str | None
    mode: _mode
    arch: _architectures


@dataclass
class Config(BaseConfig):
    setup: SetupConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    trainer: TrainerConfig

    @property
    def run_name(self) -> str:
        dataset = f"_{self.setup.dataset}"
        mode = f"_{self.setup.mode}"
        name = f"_{self.setup.name_prefix}"
        architecture = f"_{self.setup.arch}"
        return f"{NOW}_{name}{mode}{dataset}{architecture}"

    @property
    def logs_path(self) -> str:
        ckpt_path = self.setup.ckpt_path
        if ckpt_path is None:
            return str(RESULTS_PATH / self.setup.experiment_name / self.run_name / NOW)
        else:
            ckpt_path = Path(ckpt_path)
            loaded_run_path = ckpt_path.parent.parent.parent
            return str(loaded_run_path / NOW)

    @property
    def is_sppe(self) -> bool:
        return self.dataset.mode == "SPPE"

    @property
    def is_mpii(self) -> bool:
        return self.dataset.name == "MPII"

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    @property
    def num_keypoints(self) -> int:
        if self.dataset.name == "MPII":
            return 16
        elif self.dataset.name == "COCO":
            return 17
        else:
            raise ValueError("Wrong dataset name passed. Possible: ['MPII', 'COCO']")

    @property
    def hm_resolutions(self) -> list[float]:
        if self.is_sppe:
            return [1 / 4, 1 / 4]
        else:
            if self.setup.arch == "HigherHRNet":
                return [1 / 4, 1 / 2]
            elif self.setup.arch == "Hourglass":
                return [1 / 4, 1 / 4]  # for Hourglass
            else:
                raise ValueError(
                    "For MPPE mode there are only HigherHRNet and Hourglass networks available"
                )

    @property
    def ResultsClass(self) -> Type[SPPEKeypointsResults | MPPEKeypointsResults]:
        if self.is_sppe:
            if self.is_mpii:
                return SppeMpiiKeypointsResults
            else:
                return SppeCocoKeypointsResults
        else:
            if self.is_mpii:
                return MppeMpiiKeypointsResults
            else:
                return MppeCocoKeypointsResults

    @property
    def ModuleClass(self) -> Type[SPPEKeypointsModule | MPPEKeypointsModule]:
        if self.is_sppe:
            return SPPEKeypointsModule
        else:
            return MPPEKeypointsModule
