from typing import Type

from src.base.config import BaseConfig
from src.logger.pylogger import log
from src.utils.model import seed_everything


def prepare_eval_config(cfg_path: str, ckpt_path: str, ConfigClass: Type[BaseConfig]) -> BaseConfig:
    cfg = BaseConfig.from_yaml_to_dict(cfg_path)
    log.info(f"Loaded config from {cfg_path}")
    cfg["setup"]["is_train"] = False
    cfg = ConfigClass.from_dict(cfg)
    cfg.setup.ckpt_path = ckpt_path
    seed_everything(cfg.setup.seed)
    config_repr = "\n".join([f"     '{name}': {cfg}" for name, cfg in cfg.to_dict().items()])
    log.info(f"Eval config ({ConfigClass.__name__}) prepared:\n{config_repr}")
    return cfg
