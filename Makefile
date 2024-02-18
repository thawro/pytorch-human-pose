env:
	pip install poetry
	poetry install
	poetry shell

datasets:
	python src/bin/download_data.py


mlflow_server:
	cd mlflow && bash run_mlflow.sh

mlflow_example:
	cd mlflow && python test_experiment.py