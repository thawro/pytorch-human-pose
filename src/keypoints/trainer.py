from src.base.trainer import Trainer

from .datamodule import KeypointsDataModule
from .module import KeypointsModule


class KeypointsTrainer(Trainer):
    datamodule: KeypointsDataModule
    module: KeypointsModule
