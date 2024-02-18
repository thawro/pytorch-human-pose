# import logging
# import colorlog

# formatter = colorlog.ColoredFormatter(
#     fmt="%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     log_colors={
#         "DEBUG": "cyan",
#         "INFO": "green",
#         "WARNING": "yellow",
#         "ERROR": "red",
#         "CRITICAL": "bold_red",
#     },
# )

# def get_pylogger(name: str = __name__) -> logging.Logger:
#     """Initializes multi-GPU-friendly python command line logger."""
#     logger = logging.getLogger(name)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#     logger.setLevel(logging.DEBUG)
#     return logger


import logging
import colorlog
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
import time
import sys
import io
from typing import Callable


def get_cmd_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize command line logger"""
    formatter = colorlog.ColoredFormatter(
        fmt="%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_file_pylogger(filepath: str, name: str = __name__) -> logging.Logger:
    """Initialize .log file logger"""
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(filepath, "a+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger



class StdOutLogger(object):
    """Saves stdout outputs to log file."""
    def __init__(self, file_log: logging.Logger):
        self.terminal = sys.stdout
        self.log = file_log.handlers[0].stream
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)  
        
    def flush(self):
        # needed for python 3 compatibility.
        pass    
    

def remove_last_line(file_log: logging.Logger):
    """Remove the last line of log file"""
    file: io.TextIOWrapper = file_log.handlers[0].stream
    file.seek(0)
    lines = file.readlines()
    file.seek(0)
    file.truncate()
    file.writelines(lines[:-1])
    file.seek(0, 2)


def logged_tqdm(file_log: logging.Logger, tqdm_iter: tqdm_asyncio, fn: Callable, kwargs: dict):
    """Pass tqdm progressbar to log file and shot it in cmd.
    tqdm output is passed to stdout or stderr, so there is a need to pass its str form to the log file aswell,
    however logging by default appends to log files so there was a need to remove last line at each iteration, so
    the progress bar seems to be updated in the same line.
    """
    for sample in tqdm_iter:
        file_log.info(str(tqdm_iter))
        kwargs, is_break = fn(sample, **kwargs)
        remove_last_line(file_log) 
        if is_break:
            break
    file_log.info(str(tqdm_iter))
    


log = get_cmd_pylogger(__name__)

    