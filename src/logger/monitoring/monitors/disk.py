"""Class for monitoring disk stats."""

import os

import psutil

from .base import BaseMetricsMonitor


class DiskMonitor(BaseMetricsMonitor):
    """Class for monitoring disk stats."""

    def collect_metrics(self):
        # Set disk usage metrics.
        disk_usage = psutil.disk_usage(os.sep)
        self._metrics["disk_pct_used"].append(disk_usage.percent)
        self._metrics["disk_mb_used"].append(disk_usage.used / 1e6)
        self._metrics["disk_mb_available"].append(disk_usage.free / 1e6)

    def aggregate_metrics(self) -> dict[str, float]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
