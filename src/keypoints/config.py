import dataclasses
from dataclasses import dataclass
from src.utils import NOW, ROOT, RESULTS_PATH


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
    out_size: tuple[int, int]


@dataclass
class DataloaderConfig(BaseConfig):
    batch_size: int
    transform: TransformConfig


@dataclass
class SetupConfig(BaseConfig):
    experiment_name: str
    seed: int
    device: str
    dataset: str
    max_epochs: int
    limit_batches: int
    log_every_n_steps: int
    ckpt_path: str | None
    mode: str
    arch: str

    @property
    def num_keypoints(self) -> int:
        if self.dataset == "MPII":
            return 16
        elif self.dataset == "COCO":
            return 17
        else:
            raise ValueError("Wrong ds name")

    @property
    def is_sppe(self) -> bool:
        return self.mode == "SPPE"

    @property
    def is_debug(self) -> bool:
        return self.limit_batches > 0


@dataclass
class OptimizerConfig(BaseConfig):
    lr: float


@dataclass
class Config(BaseConfig):
    setup: SetupConfig
    dataloader: DataloaderConfig
    optimizer: OptimizerConfig

    @property
    def run_name(self) -> str:
        is_cont = "" if self.setup.ckpt_path is None else "_CONT"
        dataset = f"_{self.setup.dataset}"
        lr = f"_LR({self.optimizer.lr})"
        mode = f"_{self.setup.mode}"
        return f"{NOW}{is_cont}{mode}{dataset}{lr}"

    @property
    def logs_path(self):
        return str(RESULTS_PATH / self.setup.experiment_name / self.run_name)
