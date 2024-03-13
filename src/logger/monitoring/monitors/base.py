"""Base class of system metrics monitor."""

from abc import ABC, abstractmethod
from collections import defaultdict


class BaseMetricsMonitor(ABC):
    """Base class of system metrics monitor."""

    def __init__(self):
        self._metrics = defaultdict(list)

    @abstractmethod
    def collect_metrics(self):
        """Method to collect metrics.

        Subclass should implement this method to collect metrics and store in `self._metrics`.
        """
        pass

    @abstractmethod
    def aggregate_metrics(self) -> dict[str, float]:
        """Method to aggregate metrics.

        Subclass should implement this method to aggregate the metrics and return it in a dict.
        """
        pass

    @property
    def metrics(self):
        return self._metrics

    def clear_metrics(self):
        self._metrics.clear()
