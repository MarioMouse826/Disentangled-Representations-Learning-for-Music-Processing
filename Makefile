.PHONY: install test lint fmt train-scvae eval-all

install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests

fmt:
	ruff format src tests

train-scvae:
	python -m src.training.cli model=sc_vae data=nsynth_bass

eval-all:
	bash scripts/run_full_eval.sh
