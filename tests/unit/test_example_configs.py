"""The committed example configs load through the config models and reference each other."""

from __future__ import annotations

import pytest
import yaml

from rail_vision_bench.agents.prompts import load_prompt
from rail_vision_bench.config import (
    PROVIDER_NAMES,
    Mode,
    TrackerKind,
    load_model_catalogue,
    load_run_config,
    load_task_config,
    resolve_run,
)
from rail_vision_bench.eval.metrics import METRIC_NAMES
from rail_vision_bench.providers.registry import get_provider
from rail_vision_bench.schema.models import SourceKind
from rail_vision_bench.settings import Settings
from tests.conftest import REPO_ROOT

CONFIGS = REPO_ROOT / "configs"
COMPOSE = REPO_ROOT / "docker" / "compose.yaml"


@pytest.fixture
def in_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # RunConfig paths are resolved relative to the working directory, as `make dry-run` does.
    monkeypatch.chdir(REPO_ROOT)


def test_run_example_loads_and_resolves(in_repo_root: None):
    run = load_run_config(CONFIGS / "run.example.yaml")
    assert run.name == "smoke"
    assert run.mode is Mode.SINGLE_SHOT
    assert run.tracker is TrackerKind.NULL
    assert (run.limit, run.seed, run.max_rounds) == (5, 0, 3)

    resolved, task, catalogue = resolve_run(run)
    assert resolved == run
    assert task.name == "panel_topology"
    assert set(run.models) <= set(catalogue.names())


def test_catalogue_covers_every_provider_exactly_once():
    catalogue = load_model_catalogue(CONFIGS / "models.yaml")
    providers = [model.provider for model in catalogue.models]
    assert sorted(providers) == sorted(PROVIDER_NAMES)
    assert len(providers) == len(set(providers)) == len(PROVIDER_NAMES)
    names = catalogue.names()
    assert len(names) == len(set(names))
    for model in catalogue.models:
        assert model.model_id
        assert model.cost_per_1k_in is not None
        assert model.cost_per_1k_out is not None


def test_every_catalogue_provider_resolves_through_the_registry():
    catalogue = load_model_catalogue(CONFIGS / "models.yaml")
    settings = Settings(_env_file=None)
    for model in catalogue.models:
        provider = get_provider(model.provider, settings)
        assert provider.name == model.provider


def test_task_metrics_and_prompt_are_known():
    task = load_task_config(CONFIGS / "tasks" / "panel_topology.yaml")
    assert task.source_kind is SourceKind.PANEL_PHOTO
    assert task.split == "synthetic_clean"
    assert task.metrics
    assert len(task.metrics) == len(set(task.metrics))
    assert set(task.metrics) <= set(METRIC_NAMES)
    assert load_prompt(task.prompt).strip()


def test_compose_file_has_only_the_two_services():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert compose["name"] == "rail-vision-bench"
    assert set(compose["services"]) == {"n8n", "mlflow"}
    assert compose["services"]["n8n"]["image"].startswith("n8nio/n8n:")
    assert compose["services"]["mlflow"]["image"].startswith("ghcr.io/mlflow/mlflow:v")
    assert set(compose["volumes"]) == {"n8n_data", "mlflow_data"}
