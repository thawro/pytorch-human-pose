"""Loggers for training logging."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import mlflow
import yaml

from .pylogger import get_pylogger

log = get_pylogger(__name__)


class LoggerResults:
    """Storage for training results."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.steps: dict[str, list[int]] = defaultdict(lambda: [], {})
        self.metrics: dict[str, list[float]] = defaultdict(lambda: [], {})
        self.params: dict[str, Any] = {}

    def update_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Append new metrics."""
        for name, value in metrics.items():
            self.metrics[name].append(value)
            if step is not None:
                self.steps[name].append(step)

    def update_params(self, params: dict[str, Any]) -> None:
        """Update params dictionary."""
        self.params.update(params)

    def get_metrics(self) -> dict[str, dict[str, list[int | float]]]:
        """Return metrics for each split and each step."""
        metrics: dict[str, dict] = {name: {} for name in self.metrics}
        for name in self.metrics:
            metrics[name]["value"] = self.metrics[name]
            if name in self.steps:
                metrics[name]["step"] = self.steps[name]
        return metrics


class BaseLogger:
    """Base logger class."""

    def __init__(self, log_path: str | Path = "results", config: dict[str, Any] = {}):
        self.log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.ckpt_dir = self.log_path / "checkpoints"
        self.model_dir = self.log_path / "model"
        self.model_summary_dir = self.log_path / "model/summary"
        self.model_onnx_dir = self.log_path / "model/onnx"
        self.steps_examples_dir = self.log_path / "steps_examples"
        self.epochs_examples_dir = self.log_path / "epochs_examples"
        for split in ["train", "val", "test"]:
            (self.steps_examples_dir / split).mkdir(parents=True, exist_ok=True)
            (self.epochs_examples_dir / split).mkdir(parents=True, exist_ok=True)

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_summary_dir.mkdir(parents=True, exist_ok=True)
        self.model_onnx_dir.mkdir(parents=True, exist_ok=True)
        self.results = LoggerResults(config=config)
        self.log_config()

    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters as params."""
        self.log_params(params)

    def log(self, key: str, value: float, step: int | None = None) -> None:
        """Log single metric."""
        self.results.update_metrics({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics."""
        self.results.update_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self.results.update_params(params)

    def log_dict(self, dct: dict[str, Any], filename: str = "dct.yaml") -> None:
        """Log dict to yaml file."""
        path = str(self.log_path / filename)
        with open(path, "w") as yaml_file:
            yaml.dump(dct, yaml_file, default_flow_style=False)

    def log_config(self) -> None:
        """Log config to yaml."""
        self.log_dict(self.results.config, "config.yaml")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log artifact."""
        pass

    def log_checkpoints(self) -> None:
        """Log model checkpoints."""
        path = str(self.log_path / "checkpoints")
        self.log_artifact(path)

    def log_experiment(self) -> None:
        """Log experiment metrics and checkpoints."""
        log.info("Logging experiment")
        metrics = self.results.get_metrics()
        self.log_dict(metrics, "metrics.yaml")
        self.log_checkpoints()
        self.finalize(status="finished")

    def finalize(self, status: Literal["success", "failed", "finished"]) -> None:
        """Close logger"""
        log.info(f"Experiment {status}. Closing {self.__class__.__name__}")


class TerminalLogger(BaseLogger):
    """Terminal Logger (only prints info and saves locally)."""

    def log(self, key: str, value: float, step: int | None = None) -> None:
        super().log(key, value, step)
        log.info(f"Step {step}, {key}: {value}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        # log.info(f"Step {step}, Metrics: {metrics}")

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        log.info(f"Params: {params}")


class MLFlowLogger(BaseLogger):
    """Logger for logging with MLFlow."""

    def __init__(
        self,
        log_path: str | Path,
        config: dict[str, Any],
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        run_id: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        if tracking_uri is None:
            tracking_uri = "http://0.0.0.0:5000"
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            run_name=run_name,
            description=None,
        )
        super().__init__(log_path=log_path, config=config)

    @property
    def run(self) -> mlflow.ActiveRun:
        return mlflow.active_run()

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def log(self, key: str, value: float, step: int | None = None) -> None:
        super().log(key, value, step)
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        mlflow.log_params(params)

    def log_dict(self, config: dict[str, Any], filename: str = "config.yaml") -> None:
        super().log_dict(config, filename)
        mlflow.log_dict(config, filename)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def download_artifact(
        self,
        artifact_path: str,
        run_id: str | None = None,
    ) -> str:
        """Download artifact from mlflow.
        Artifact is stored in dst_path (relative to log dir) directory
        Returns path to the downloaded artifact
        """
        dst_path = str(self.log_path / "loaded")
        if run_id is None:
            run_id = self.run_id
        log.info(
            f"Downloading {artifact_path} from mlflow run {self.run_id} to {dst_path}"
        )
        return mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
        )

    def finalize(self, status: Literal["success", "failed", "finished"]) -> None:
        super().finalize(status)
        mlflow.end_run(status=status)
