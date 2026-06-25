#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = restaurant-visitor-eda
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies from uv.lock
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf site/ dist/ build/ *.egg-info/
	@echo ">>> Project cleaned up successfully!"

## Format source code with ruff (modifies files)
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Run all pre-commit hooks (lint, format, type-check) on all files
.PHONY: lint
lint:
	uv run pre-commit run --all-files

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> source .venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run data processing script
.PHONY: data
data: requirements
	uv run python -m restaurant_visitor_eda.dataset
	uv run python -m restaurant_visitor_eda.features

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

## Train the final model using best parameters
.PHONY: train
train:
	uv run python -m restaurant_visitor_eda.modeling.train

## Generate submission using the trained model
.PHONY: predict
predict:
	uv run python -m restaurant_visitor_eda.modeling.predict

## Run the full end-to-end pipeline
.PHONY: pipeline
pipeline: data train predict
	@echo ">>> End-to-end pipeline completed successfully! Submission file is ready."

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
