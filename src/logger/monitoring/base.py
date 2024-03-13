"""Class for monitoring system stats."""

import threading
from abc import ABC, abstractmethod

from src.logger.pylogger import log


class BaseSystemMonitor(ABC):
    """Base class for System Monitors"""

    def __init__(self, sampling_interval: float = 10):
        self.sampling_interval = sampling_interval
        self._shutdown_event = threading.Event()
        self._process = None

    def start(self):
        """Start monitoring system metrics."""
        name = self.__class__.__name__
        try:
            self._process = threading.Thread(
                target=self._monitor,
                daemon=True,
                name=name,
            )
            self._process.start()
            log.info(f"Started {name}.")
        except Exception as e:
            log.warning(f"Failed to start {name}: {e}")
            self._process = None

    def _monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        while not self._shutdown_event.is_set():
            self.monitor()

    @abstractmethod
    def monitor(self):
        raise NotImplementedError()

    def finish(self):
        """Stop monitoring system metrics."""
        name = self.__class__.__name__
        if self._process is None:
            return
        log.info(f"..Stopping {name}..")
        self._shutdown_event.set()
        try:
            self._process.join()
            log.info(f"     Successfully terminated {name}!")
        except Exception as e:
            log.exception(f"Error terminating {name} process: {e}.")
        self._process = None
