from collections import defaultdict
from itertools import groupby
from typing import Literal

import torch

from src.logger.pylogger import log

_metric = dict[str, list[dict[str, int | float]]]
_metrics = dict[str, _metric]


class MetricsStorage:
    metrics: _metrics

    def __init__(self, name: str = "", metrics: _metrics | None = None) -> None:
        if metrics is None:
            metrics = defaultdict(lambda: defaultdict(lambda: [], {}))
        self.metrics = metrics
        self.name = name

    @property
    def logged_metrics(self) -> list[str]:
        _logged_metrics = []
        for metric_name, splits_metrics in self.metrics.items():
            for split_name, values in splits_metrics.items():
                if len(values) > 0:
                    _logged_metrics.append(metric_name)
                    break
        return _logged_metrics

    def clear(self):
        self.metrics = defaultdict(lambda: defaultdict(lambda: [], {}))

    def get(self, metric_name: str, stage: str) -> list[dict]:
        return self.metrics[metric_name][stage]

    def aggregate_over_key(self, key: Literal["epoch", "step"]) -> "MetricsStorage":
        epochs_metrics = {}
        for metric_name, splits_metrics in self.metrics.items():
            epochs_metrics[metric_name] = {}
            for split_name, logged_values in splits_metrics.items():
                key_fn = lambda v: v[key]
                logged_values = sorted(logged_values, key=key_fn)
                key_aggregated_values = []
                for key_step, grouped in groupby(logged_values, key_fn):
                    grouped = list(grouped)
                    values = [el["value"] for el in grouped]
                    avg_value = sum(values) / len(values)
                    agg_values = {key: key_step, "value": avg_value}
                    if key != "epoch":
                        agg_values["epoch"] = grouped[0]["epoch"]
                    key_aggregated_values.append(agg_values)
                epochs_metrics[metric_name][split_name] = key_aggregated_values
        return MetricsStorage(name=key, metrics=epochs_metrics)

    def inverse_nest(self) -> _metrics:
        inverse_metrics = defaultdict(lambda: defaultdict(lambda: [], {}))
        for metric_name, splits_metrics in self.metrics.items():
            for split_name, values in splits_metrics.items():
                inverse_metrics[split_name][metric_name] = values
        return inverse_metrics

    def __getitem__(self, metric_name: str) -> _metric:
        return self.metrics[metric_name]

    def append(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            values = {"step": step, "value": value, "epoch": epoch}
            if metric not in self.metrics:
                self.metrics[metric] = {}
            if split not in self.metrics[metric]:
                self.metrics[metric][split] = []
            self.metrics[metric][split].append(values)

    def to_dict(self) -> dict:
        """For state saving"""
        _metrics = {}
        for metric_name, splits_metrics in self.metrics.items():
            _metrics[metric_name] = {}
            for split_name, values in splits_metrics.items():
                _metrics[metric_name][split_name] = values
        return _metrics

    def state_dict(self) -> dict:
        return {"metrics": self.to_dict()}

    def load_state_dict(self, state_dict: dict):
        self.metrics = state_dict["metrics"]
        log.info(f'     Loaded "{self.name}" metrics state')


class SystemMonitoringStorage:
    def __init__(self) -> None:
        self.metrics = defaultdict(list)
        self.timestamps = []

    def update(self, metrics: dict[str, float], timestamp: str):
        for k, v in metrics.items():
            self.metrics[k].append(v)
        self.timestamps.append(timestamp)
