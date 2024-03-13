import os

from src.logger.pylogger import log

from .base import BaseSystemMonitor


class NvidiaSmiMonitor(BaseSystemMonitor):
    """Log nvidia-smi output to log file.

    ..Example::
        import time
        from src.logger.monitoring.nvidia_smi import NvidiaSmiMonitor

        monitor = NvidiaSmiMonitor("test.log", sampling_interval=1)
        monitor.start()
        time.sleep(15)
    """

    def __init__(self, log_filepath: str, sampling_interval: float = 1):
        super().__init__(sampling_interval)
        self.log_filepath = log_filepath
        log.info(
            f"..Initializing nvidia-smi output logging (every {self.sampling_interval} seconds) to '{self.log_filepath}'.."
        )

    def monitor(self):
        """Log nvidia-smi and gpustat outputs to log file."""
        command = (
            f"nvidia-smi > {self.log_filepath} ; "
            f"echo '\n\ngpustat (https://github.com/wookayin/gpustat) summary:\n' >> {self.log_filepath} ; "
            f"gpustat -c -p -F -P >> {self.log_filepath}"
        )
        os.system(command)
        self._shutdown_event.wait(self.sampling_interval)

    def finish(self):
        """Stop monitoring system metrics."""
        if self._process is None:
            return
        log.info("Stopping system metrics monitoring...")
        self._shutdown_event.set()
        try:
            self._process.join()
            log.info("Successfully terminated system metrics monitoring!")
        except Exception as e:
            log.error(f"Error terminating system metrics monitoring process: {e}.")
        self._process = None
