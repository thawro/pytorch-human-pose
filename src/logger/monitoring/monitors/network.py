"""Class for monitoring network stats."""

import psutil

from .base import BaseMetricsMonitor


class NetworkMonitor(BaseMetricsMonitor):
    def __init__(self):
        super().__init__()
        self._set_initial_metrics()
        self._metrics: dict[str, float] = {}

    def _set_initial_metrics(self):
        # Set initial network usage metrics. `psutil.net_io_counters()` counts the stats since the
        # system boot, so to set network usage metrics as 0 when we start logging, we need to keep
        # the initial network usage metrics.
        network_usage = psutil.net_io_counters()
        self._initial_receive_megabytes = network_usage.bytes_recv / 1e6
        self._initial_transmit_megabytes = network_usage.bytes_sent / 1e6

    def collect_metrics(self):
        # Set network usage metrics.
        network_usage = psutil.net_io_counters()
        # Usage metrics will be the diff between current and initial metrics.
        self._metrics["network_mb_receive"] = (
            network_usage.bytes_recv / 1e6 - self._initial_receive_megabytes
        )
        self._metrics["network_mb_transmit"] = (
            network_usage.bytes_sent / 1e6 - self._initial_transmit_megabytes
        )

    def aggregate_metrics(self) -> dict[str, float]:
        # Network metrics don't need to be averaged.
        return dict(self._metrics)
