# Developer and CI entry points. Every CI step is a target here so local runs and CI cannot drift.
# Override the interpreter or venv on the command line: make test PY=3.11 VENV=/path/to/venv311

PY ?= 3.12
VENV ?= .venv
BIN := $(abspath $(VENV))/bin
# makes every `uv pip ...` call target the project venv, whether VENV is relative or absolute
export VIRTUAL_ENV := $(abspath $(VENV))
# the ONLY sanctioned way to (re)generate the lock: an in-place compile keeps existing pins
LOCK_CMD := uv pip compile requirements.txt -o requirements.lock --universal --python-version 3.11

.DEFAULT_GOAL := check
.PHONY: help venv install-ci install lock lock-check lint format typecheck test test-cov \
	schema schema-check validate-examples dry-run check build-check serve synth \
	compose-up compose-down pre-commit clean

help:  ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F ':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

venv:  ## create the venv (idempotent)
	uv venv --python $(PY) --allow-existing $(VENV)

install-ci: venv  ## install the lock, the editable package and the typing stubs
	uv pip sync requirements.lock
	uv pip install --no-deps -e .
	uv pip install --group typing

install: install-ci  ## install-ci plus the git hooks
	$(BIN)/pre-commit install

lock:  ## regenerate requirements.lock in place
	$(LOCK_CMD)

lock-check:  ## fail if requirements.lock is out of sync with requirements.txt
	$(LOCK_CMD)
	git diff --exit-code -- requirements.lock

lint:  ## ruff check + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## ruff format + autofix
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:  ## mypy (config in pyproject.toml)
	$(BIN)/mypy

test:  ## pytest
	$(BIN)/pytest

test-cov:  ## pytest with coverage (coverage.xml for CI)
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=xml

schema:  ## regenerate schema/v0.json from the pydantic models
	$(BIN)/bench schema export --out schema/v0.json

schema-check:  ## fail if schema/v0.json drifted from the models
	$(BIN)/bench schema export --check

validate-examples:  ## strict-validate the committed example documents
	$(BIN)/bench validate --strict schema/examples/v0/*.json

dry-run:  ## resolve the example run config without running anything
	$(BIN)/bench run --config configs/run.example.yaml --dry-run

check: lint typecheck schema-check validate-examples test  ## everything CI runs per interpreter

build-check:  ## build the wheel and verify dynamic deps + package data landed in it
	rm -rf dist
	uv build
	unzip -p dist/*.whl '*/METADATA' | grep -q '^Requires-Dist: pydantic>=2.9'
	$(BIN)/python -m zipfile -l dist/*.whl | grep -q 'rail_vision_bench/py.typed'
	$(BIN)/python -m zipfile -l dist/*.whl | grep -q 'prompts/single_shot.md'

serve:  ## run the real-time harness with reload
	$(BIN)/bench serve --reload

synth:  ## generate synthetic data (exits 3 in the skeleton)
	$(BIN)/bench synth generate --out data/synthetic --n 10 --seed 0

compose-up:  ## start n8n + MLflow
	docker compose -f docker/compose.yaml up -d

compose-down:  ## stop n8n + MLflow
	docker compose -f docker/compose.yaml down

pre-commit:  ## run every hook on the whole tree
	$(BIN)/pre-commit run --all-files

clean:  ## remove tool caches and build output (never data/ or runs/)
	rm -rf .ruff_cache .mypy_cache .pytest_cache .hypothesis build dist coverage.xml .coverage htmlcov
