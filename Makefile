env:
	pip install poetry
	poetry install
	poetry shell

datasets:
	python src/bin/download_data.py