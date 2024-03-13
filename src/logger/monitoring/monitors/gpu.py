"""Class for monitoring GPU stats."""

import sys

from .base import BaseMetricsMonitor

try:
    import pynvml
except ImportError:
    # If `pynvml` is not installed, a warning will be logged at monitor instantiation.
    # We don't log a warning here to avoid spamming warning at every import.
    pass


class GPUMonitor(BaseMetricsMonitor):
    """Class for monitoring GPU stats."""

    def __init__(self):
        if "pynvml" not in sys.modules:
            # Only instantiate if `pynvml` is installed.
            raise ImportError(
                "`pynvml` is not installed, to log GPU metrics please run `pip install pynvml` "
                "to install it."
            )
        try:
            # `nvmlInit()` will fail if no GPU is found.
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML, skip logging GPU metrics: {e}")

        super().__init__()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def collect_metrics(self):
        # Set GPU metrics.
        for i, handle in enumerate(self.gpu_handles):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = float(memory.used)
            total = float(memory.total)

            self._metrics[f"gpu_{i}_pct_memory_used"].append(round(used / total * 100, 1))
            self._metrics[f"gpu_{i}_mb_memory_used"].append(used / 1e6)

            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._metrics[f"gpu_{i}_pct_utilisation"].append(device_utilization.gpu)

    def aggregate_metrics(self) -> dict[str, float]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
