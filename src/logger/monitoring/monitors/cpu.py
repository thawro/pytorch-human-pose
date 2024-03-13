"""Class for monitoring CPU stats."""

import psutil

from .base import BaseMetricsMonitor


class CPUMonitor(BaseMetricsMonitor):
    """Class for monitoring CPU stats."""

    def collect_metrics(self):
        # Set CPU metrics.
        cpu_percent = psutil.cpu_percent()
        self._metrics["cpu_pct_utilisation"].append(cpu_percent)

        system_memory = psutil.virtual_memory()
        used = system_memory.used
        total = system_memory.total
        self._metrics["system_mb_memory_used"].append(used / 1e6)
        self._metrics["system_pct_memory_used"].append(used / total * 100)

    def aggregate_metrics(self) -> dict[str, float]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
