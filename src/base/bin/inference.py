from typing import Type

from src.base.config import BaseConfig
from src.logger.pylogger import log
from src.utils.model import seed_everything


def prepare_inference_config(cfg_path: str, ConfigClass: Type[BaseConfig]) -> BaseConfig:
    cfg = BaseConfig.from_yaml_to_dict(cfg_path)
    cfg["setup"]["is_train"] = False
    cfg = ConfigClass.from_dict(cfg)
    cfg.setup.ckpt_path = cfg.inference.ckpt_path
    seed_everything(cfg.setup.seed)
    log.info(f"Inference config ({ConfigClass.__name__}) prepared.")
    log.info(f"Inference settings:\n{cfg.inference}")
    return cfg
