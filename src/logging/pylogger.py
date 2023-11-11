"""Colorful Logger used for terminal logging"""

import logging

import colorlog

formatter = colorlog.ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger
