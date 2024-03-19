"""Class for monitoring system stats."""

from typing import Any, Protocol

from src.logger.pylogger import log
from src.utils.utils import get_current_date_and_time

from .base import BaseSystemMonitor
from .monitors.cpu import CPUMonitor
from .monitors.disk import DiskMonitor
from .monitors.gpu import GPUMonitor
from .monitors.network import NetworkMonitor


class MetricsCallbackProtocol(Protocol):
    """Do something with metrics, e.g. log it using custom logger"""

    def __call__(self, metrics: dict[str, float], timestamp: str) -> Any: ...


class SystemMetricsMonitor(BaseSystemMonitor):
    """Class for monitoring system stats.

    This class is used for pulling system metrics. Calling `start()` will
    spawn a thread that logs system metrics periodically. Calling `finish()` will stop the thread.
    Logging is done on a different frequency from pulling metrics, so that the metrics are
    aggregated over the period.

    ..Example:
        import time
        from src.logger.monitoring.system import SystemMetricsMonitor

        def print_metrics_callback(metrics: dict[str, float], timestamp: str):
            print(timestamp, metrics)

        monitor = SystemMetricsMonitor(
            sampling_interval=1,
            metrics_callback=print_metrics_callback
        )
        monitor.start()

        time.sleep(15)

        Args:
            sampling_interval: float, default to 10. The interval (in seconds) at which to pull system
                metrics
            samples_before_logging: int, default to 1. The number of samples to aggregate before
                logging.
    """

    def __init__(
        self,
        sampling_interval=10,
        samples_before_logging=1,
        metrics_callback: MetricsCallbackProtocol | None = None,
    ):
        super().__init__(sampling_interval)
        self.monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor()]
        try:
            gpu_monitor = GPUMonitor()
            self.monitors.append(gpu_monitor)
        except Exception as e:
            log.warning(
                f"Skip logging GPU metrics because creating `GPUMonitor` failed with error: {e}."
            )
        self.samples_before_logging = samples_before_logging
        self.metrics_callback = metrics_callback

    def monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        for _ in range(self.samples_before_logging):
            self.collect_metrics()
            self._shutdown_event.wait(self.sampling_interval)
        metrics = self.aggregate_metrics()
        try:
            self.publish_metrics(metrics)
        except Exception as e:
            log.exception(f"Failed to log system metrics: {e}")
            return

    def collect_metrics(self):
        """Collect system metrics."""
        metrics = {}
        for monitor in self.monitors:
            monitor.collect_metrics()
            metrics.update(monitor._metrics)
        return metrics

    def aggregate_metrics(self):
        """Aggregate collected metrics."""
        metrics = {}
        for monitor in self.monitors:
            metrics.update(monitor.aggregate_metrics())
        return metrics

    def publish_metrics(self, metrics: dict[str, float]):
        """Do something with collected metrics and clear them."""
        timestamp = get_current_date_and_time(format="%y-%m-%d %H:%M:%S")
        if self.metrics_callback is not None:
            self.metrics_callback(metrics, timestamp)
        for monitor in self.monitors:
            monitor.clear_metrics()
