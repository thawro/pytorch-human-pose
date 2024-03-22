env:
	bash scripts/prepare_env.sh

dirs:
	bash scripts/prepare_dirs.sh

imagenet:
	bash scripts/prepare_imagenet.sh

coco:
	bash scripts/prepare_coco.sh

save_coco_annots:
	poetry shell && python scripts/save_coco_annots.py

mlflow_server:
	bash scripts/run_mlflow.sh

mlflow_example:
	cd mlflow && python test_experiment.py

gpu_powerlimit:
	sudo bash scripts/set_gpu_powerlimit.sh