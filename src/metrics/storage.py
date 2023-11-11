from collections import defaultdict
import torch


class MetricsStorage:
    metrics: dict[str, dict[str, list]]

    def __init__(self) -> None:
        self.metrics = defaultdict(lambda: defaultdict(lambda: [], {}))

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

    def get(self, metric_name: str, stage: str) -> list:
        return self.metrics[metric_name][stage]

    def inverse_nest(self) -> dict[str, dict[str, list]]:
        inverse_metrics = defaultdict(lambda: defaultdict(lambda: [], {}))
        for metric_name, splits_metrics in self.metrics.items():
            for split_name, values in splits_metrics.items():
                inverse_metrics[split_name][metric_name] = values
        return inverse_metrics

    def __getitem__(self, metric_name: str) -> dict[str, list]:
        return self.metrics[metric_name]

    def append(self, metrics: dict[str, float], split: str = "") -> None:
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if metric not in self.metrics and not isinstance(self.metrics, defaultdict):
                already_logged_metrics = list(self.metrics.keys())
                if len(already_logged_metrics) > 0:
                    other_metric = already_logged_metrics[0]
                    splits = list(self.metrics[other_metric].keys())
                    self.metrics[metric] = {
                        _split: [0] * len(self.metrics[other_metric][_split])
                        for _split in splits
                    }
                else:
                    self.metrics = defaultdict(lambda: defaultdict(lambda: [], {}))

            self.metrics[metric][split].append(value)

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
