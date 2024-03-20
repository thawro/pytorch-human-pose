mlflow_server:
	cd mlflow && bash run_mlflow.sh

mlflow_example:
	cd mlflow && python test_experiment.py