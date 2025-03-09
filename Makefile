# Oneshell means one can run multiple lines in a recipe in the same shell, so one doesn't have to
# chain commands together with semicolon
.ONESHELL:
SHELL=/bin/bash
PACKAGE=thesis_work
PRECOMMIT_FILE_PATHS=./thesis_work/__init__.py
PROFILE_FILE_PATH=./thesis_work/__init__.py
DOCKER_IMAGE=thesis-work
DOCKER_TARGET=development

# Add timeout for uv operations (1800 seconds)
export UV_HTTP_TIMEOUT=1800

.PHONY: help install pre-commit format lint docker gui
.DEFAULT_GOAL=help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

# If .env file exists, include it and export its variables
ifeq ($(shell test -f .env && echo 1),1)
    include .env
	export
endif

python-info: ## List information about the python environment
	@which uv run python
	uv run python --version

install-uv: ## Install uv
	! command -v uv &> /dev/null && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$$HOME/.local/bin" sh
	source $$HOME/.local/bin/env bash

update-uv: ## Update uv to the latest version
	@uv self update

install-base: ## Installs only package dependencies
	uv sync --frozen --no-dev --no-install-project

install: ## Installs the development version of the package
	$(MAKE) install-uv
	$(MAKE) update-uv
	@uv sync --frozen
	$(MAKE) install-precommit

install-no-cache: ## Installs the development version of the package without cache
	$(MAKE) install-uv
	$(MAKE) update-uv
	uv sync --frozen --no-cache
	$(MAKE) install-precommit

install-precommit: ## Install pre-commit hooks
	uv run pre-commit install

update-dependencies: ## Updates the lockfiles and installs dependencies. Dependencies are updated if necessary
	uv sync

upgrade-dependencies: ## Updates the lockfiles and installs the latest version of the dependencies
	uv sync -U

pre-commit-one: ## Run pre-commit with specific files
	uv lock --locked
	uv run pre-commit run --files ${PRECOMMIT_FILE_PATHS}

pre-commit: ## Run pre-commit for all package files
	uv lock --locked
	uv run pre-commit run --all-files

pre-commit-clean: ## Clean pre-commit cache
	uv run pre-commit clean

lint: ## Lint code with ruff
	uv lock --locked
	uv run --module ruff format ${PACKAGE} --check --diff
	uv run --module ruff check ${PACKAGE}

lint-report: ## Lint report for gitlab
	uv lock --locked
	uv run --module ruff format ${PACKAGE} --check --diff
	uv run --module ruff check ${PACKAGE} --format gitlab > gl-code-quality-report.json

format: ## Run ruff for all package files. CHANGES CODE
	uv lock --locked
	uv run --module ruff format ${PACKAGE}
	uv run --module ruff check ${PACKAGE} --fix --show-fixes

format-unsafe: ## Run ruff for all package files. CHANGES CODE
	uv lock --locked
	uv run --module ruff format ${PACKAGE}
	uv run --module ruff check ${PACKAGE} --fix --unsafe-fixes --show-fixes

dagster-dev:  ## Run dagster development env with environment variables
	dagster-webserver - p 3006
	# dagster dev -p 3006

docker: ## Build docker image
	docker build --tag ${DOCKER_IMAGE}:${DOCKER_TARGET} --file docker/Dockerfile --target ${DOCKER_TARGET} .

gui: ## Run GUI with streamlit
	streamlit run thesis_work/gui/index.py

docker-gpu-commands:
	# docker compose -f docker-compose_gpu.yaml build thesis-work-gpu
	# docker compose -f docker-compose_gpu.yaml up thesis-work-gpu -d
	# docker compose -f docker-compose_gpu.yaml down

	# docker exec -it <container_id> /bin/bash
