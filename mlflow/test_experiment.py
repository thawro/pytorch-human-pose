import os
import mlflow
import logging
import colorlog


def get_cmd_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize command line logger"""
    formatter = colorlog.ColoredFormatter(
        fmt="%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger


log = get_cmd_pylogger(__name__)

if __name__ == "__main__":
    log.info("START")
    HOST = "localhost"  # 127.0.0.1
    PORT = "5000"
    tracking_uri = f"http://{HOST}:{PORT}"
    log.info(f"..Setting MLFlow tracking uri to {tracking_uri}..")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "SomeExperiment"
    log.info(f"..Setting MLFlow experiment to {experiment_name}..")
    experiment = mlflow.set_experiment(experiment_name)

    log.info("..Starting run..")
    run = mlflow.start_run(run_id=None)
    run_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
    log.info(f"Visit run at: {run_url}")
    log.info("..Logging example params (param_1) and metrics (foo)..")
    mlflow.log_param("param_1", 123)

    mlflow.log_metric("foo", 3)
    mlflow.log_metric("foo", 4)
    mlflow.log_metric("foo", 5)

    log.info("..Creating local artifacts..")
    outputs_dir = "example_outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    with open(f"{outputs_dir}/test.txt", "w") as f:
        f.write("hello world!")

    log.info(f"..Loging example artifacts ({outputs_dir}) to mlflow artifacts..")
    mlflow.log_artifacts("example_outputs")
    mlflow.end_run()
    log.info("END")
