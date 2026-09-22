"""Shared pytest configuration: hypothesis profiles, repository paths, example fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, settings

from rail_vision_bench.schema.models import SceneAnnotation

# Deadlines are disabled in both profiles because validation time depends on the
# generated scene size; the CI profile is derandomized so failures are reproducible.
settings.register_profile("default", max_examples=50, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "schema" / "examples" / "v0"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document from ``path`` as a plain dictionary."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), path
    return loaded


@pytest.fixture
def minimal_doc() -> dict[str, Any]:
    """The minimal example as a raw dictionary."""
    return load_json(EXAMPLES_DIR / "minimal.json")


@pytest.fixture
def station_dkw_doc() -> dict[str, Any]:
    """The reference station example as a raw dictionary."""
    return load_json(EXAMPLES_DIR / "station_dkw.json")


@pytest.fixture
def minimal_scene(minimal_doc: dict[str, Any]) -> SceneAnnotation:
    """The minimal example as a parsed model."""
    return SceneAnnotation.model_validate(minimal_doc)


@pytest.fixture
def station_dkw_scene(station_dkw_doc: dict[str, Any]) -> SceneAnnotation:
    """The reference station example as a parsed model."""
    return SceneAnnotation.model_validate(station_dkw_doc)


# G3a adds here: the `cli` fixture (typer.testing.CliRunner) and the autouse
# `clean_env` fixture that deletes every SETTINGS_ENV_KEYS entry from the
# environment and calls get_settings.cache_clear().


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests that need a display or the network unless the environment opts in."""
    if item.get_closest_marker("requires_display") and not os.environ.get("DISPLAY"):
        pytest.skip("needs a display (DISPLAY is not set)")
    if item.get_closest_marker("requires_network") and os.environ.get("RVB_NETWORK_TESTS") != "1":
        pytest.skip("needs network access (set RVB_NETWORK_TESTS=1)")
