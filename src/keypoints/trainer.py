from src.base.trainer import Trainer

from .datamodule import KeypointsDataModule
from .module import KeypointsModule


class KeypointsTrainer(Trainer):
    datamodule: KeypointsDataModule
    module: KeypointsModule

    def on_epoch_start(self):
        pass
        # self.datamodule.train_ds.mosaic_probability = 0
