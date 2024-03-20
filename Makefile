env:
	bash scripts/prepare_env.sh

dirs:
	bash scripts/prepare_dirs.sh

imagenet:
	bash scripts/prepare_imagenet.sh

coco:
	bash scripts/prepare_coco.sh

mlflow_server:
	bash scripts/run_mlflow.sh

mlflow_example:
	cd mlflow && python test_experiment.py