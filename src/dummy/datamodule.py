"""DataModule used to load DataLoaders"""

from src.logging import get_pylogger
from src.base.datamodule import DataModule


log = get_pylogger(__name__)


class DummyDataModule(DataModule):
    pass
