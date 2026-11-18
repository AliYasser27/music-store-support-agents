.DEFAULT_GOAL := help
PYTHON ?= python3
VENV := .venv

.PHONY: help setup install lint format test test-all demo notebook clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install everything
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt

lint: ## Static checks (ruff)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format: ## Auto-fix and format code
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

test: ## Run the offline test suite (no API key needed)
	$(VENV)/bin/pytest -m "not e2e"

test-all: ## Run every test, including the live LLM end-to-end tests
	$(VENV)/bin/pytest

eval: ## Supervisor routing accuracy against the labelled set (needs GROQ_API_KEY)
	$(VENV)/bin/python -m evaluation.routing_eval

demo: ## Run the scripted demo (needs GROQ_API_KEY)
	$(VENV)/bin/python -m src.run --demo

notebook: ## Open the walkthrough notebook
	$(VENV)/bin/jupyter lab music_store_customer_support.ipynb

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .ipynb_checkpoints
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
