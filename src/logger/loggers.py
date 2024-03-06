"""Loggers for training logging."""

import logging
import uuid
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

import mlflow
from src.utils.config import LOG_DEVICE_ID
from src.utils.utils import get_rank

from .pylogger import log


class Status(Enum):
    """Based on MLFlow"""

    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class LoggerResults:
    """Storage for training results."""

    def __init__(self):
        self.steps: dict[str, list[int]] = defaultdict(lambda: [], {})
        self.metrics: dict[str, list[float]] = defaultdict(lambda: [], {})
        self.params: dict[str, Any] = {}

    def update_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
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

    name: str = "base"

    def __init__(self, log_path: str | Path = "results", config: dict[str, Any] = {}):
        self.config = config
        self._log_config_info()
        self.log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.logs_path = self.log_path / "logs"
        self.ckpt_dir = self.log_path / "checkpoints"
        self.model_dir = self.log_path / "model"
        self.model_summary_dir = self.log_path / "model/summary"
        self.model_onnx_dir = self.log_path / "model/onnx"
        self.eval_examples_dir = self.log_path / "eval_examples"
        self.data_examples_dir = self.log_path / "data_examples"
        # creating directories
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_summary_dir.mkdir(parents=True, exist_ok=True)
        self.model_onnx_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(exist_ok=True, parents=True)
        self.eval_examples_dir.mkdir(exist_ok=True, parents=True)
        self.data_examples_dir.mkdir(exist_ok=True, parents=True)
        self.results = LoggerResults()

        self._run_id = None

    def _log_config_info(self):
        rank = get_rank()
        if rank != 0:
            return
        config_repr = "\n".join([f"     '{name}': {cfg}" for name, cfg in self.config.items()])
        log.info(f"Experiment config:\n{config_repr}")

    @property
    def run_id(self) -> str:
        if self._run_id is not None:
            return self._run_id
        else:
            self._run_id = str(uuid.uuid1())
            return self._run_id

    def start_run(self):
        log.info(f"..Starting {self.__class__.__name__} run")

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
        self.log_dict(self.config, "config.yaml")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log artifact."""
        pass

    def finalize(self, status: Status) -> None:
        """Close logger"""
        log.warn(f"Experiment {status.value}. Closing {self.__class__.__name__}")


class Loggers:
    """Class to be used in Trainer"""

    def __init__(self, loggers: list[BaseLogger], device_id: int, file_log: logging.Logger):
        # make sure that only device at LOG_DEVICE_ID is logging
        if device_id != LOG_DEVICE_ID:
            loggers = []
        self.loggers = loggers
        self.device_id = device_id
        self.file_log = file_log

    def start_run(self):
        for logger in self.loggers:
            logger.start_run()
            logger.log_config()

    def log(self, key: str, value: float, step: int | None = None):
        for logger in self.loggers:
            logger.log(key=key, value=value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        for logger in self.loggers:
            logger.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        for logger in self.loggers:
            logger.log_params(params=params)

    def log_dict(self, dct: dict[str, Any], filename: str = "dct.yaml"):
        for logger in self.loggers:
            logger.log_dict(dct=dct, filename=filename)

    def log_config(self):
        for logger in self.loggers:
            logger.log_config()

    def log_artifact(self, local_path: str, artifact_path: str):
        for logger in self.loggers:
            logger.log_artifact(local_path, artifact_path)

    def finalize(self, status: Status):
        for logger in self.loggers:
            logger.finalize(status)

    def state_dict(self) -> dict:
        run_ids = {}
        for logger in self.loggers:
            run_ids[logger.name] = logger.run_id
        return run_ids


class TerminalLogger(BaseLogger):
    name: str = "terminal"
    """Terminal Logger (only prints info and saves locally)."""

    def log(self, key: str, value: float, step: int | None = None) -> None:
        super().log(key, value, step)
        log.info(f"Step {step}, {key}: {value}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        log.info(f"Params: {params}")


class MLFlowLogger(BaseLogger):
    """Logger for logging with MLFlow."""

    client: mlflow.client.MlflowClient
    run: mlflow.entities.Run
    name: str = "mlflow"

    def __init__(
        self,
        log_path: str | Path,
        config: dict[str, Any],
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        run_id: str | None = None,
        resume: bool = True,
        description: str = "",
    ):
        super().__init__(log_path=log_path, config=config)
        if tracking_uri is None:
            HOST = "localhost"  # 127.0.0.1
            PORT = "5000"
            tracking_uri = f"http://{HOST}:{PORT}"
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.resume = resume
        self.description = description
        self._run_id = run_id

    def start_run(self):
        super().start_run()
        client = mlflow.client.MlflowClient(self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(self.experiment_name)
            experiment = client.get_experiment(experiment_id)
        experiment_id = experiment.experiment_id
        if not self.resume:
            log.info(f"     Creating new run with {self.run_name} name")
            run = client.create_run(experiment_id, run_name=self.run_name)
        elif self._run_id is None:
            # get run by name
            runs = client.search_runs(
                experiment_ids=[str(experiment_id)],
                filter_string=f'tags.mlflow.runName = "{self.run_name}"',
            )
            num_runs = len(runs)
            if num_runs == 0:
                log.info(
                    f"     There is no run with {self.run_name} name on mlflow server (for experiment '{self.experiment_name}')"
                )
                log.info(f"     Creating new run with {self.run_name} name")
                run = client.create_run(experiment_id, run_name=self.run_name)
            if num_runs == 1:
                log.info(f"     Found existing run with {self.run_name} name on mlflow server")
                run = runs[0]
                log.info(f"     Resuming Run {run.info.run_name} (ID = {run.info.run_id})")
            elif num_runs > 1:
                log.warn(
                    f"     More than one run with {self.run_name} name found on mlflow server. Raising Exception"
                )
                e = ValueError()
                log.exception(e)
                raise e
        else:
            try:
                run = client.get_run(self._run_id)  # get run by id
            except Exception as e:
                log.exception(e)
                raise e
        self.client = client
        self.run = run
        run_url = (
            f"{self.tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
        )
        log.info(f"     Visit run at: {run_url}")

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def log(self, key: str, value: float, step: int | None = None) -> None:
        super().log(key, value, step)
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        for key, value in metrics.items():
            self.log(key, value, step=step)

    def log_param(self, key: str, value: float | str | int) -> None:
        self.client.log_param(self.run_id, key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_dict(self, config: dict[str, Any], filename: str = "config.yaml") -> None:
        super().log_dict(config, filename)
        self.client.log_dict(self.run_id, config, filename)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.client.log_artifacts(self.run_id, local_path, artifact_path)

    def download_artifact(self, artifact_path: str) -> str:
        """Download artifact from mlflow.
        Artifact is stored in dst_path (relative to log dir) directory
        Returns path to the downloaded artifact
        """
        dst_path = str(self.log_path / "loaded")
        log.info(f"Downloading {artifact_path} from mlflow run {self.run_id} to {dst_path}")
        return self.client.download_artifacts(
            run_id=self.run_id,
            artifact_path=artifact_path,
            dst_path=dst_path,
        )

    def finalize(self, status: Status) -> None:
        super().finalize(status)
        self.log_artifact(str(self.logs_path), "logs")
        self.client.set_terminated(self.run_id, status=status.value)
