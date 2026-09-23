"""Config models, YAML loaders, overrides, run resolution and the run-directory contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from pydantic import ValidationError

from rail_vision_bench.config import (
    PROVIDER_NAMES,
    Mode,
    ModelCatalogue,
    ModelConfig,
    ProviderName,
    RunConfig,
    TaskConfig,
    TrackerKind,
    apply_overrides,
    load_model_catalogue,
    load_run_config,
    load_task_config,
    load_yaml,
    resolve_run,
)
from rail_vision_bench.runner import RUN_LAYOUT, run_dir
from rail_vision_bench.schema.models import SourceKind

CATALOGUE: dict[str, Any] = {
    "models": [
        {"name": "m1", "provider": "ollama", "model_id": "qwen2.5vl:7b"},
        {
            "name": "m2",
            "provider": "anthropic",
            "model_id": "claude-x",
            "supports_tools": True,
            "cost_per_1k_in": 0.003,
            "cost_per_1k_out": 0.015,
            "params": {"max_tokens": 2048},
        },
    ]
}
TASK: dict[str, Any] = {
    "name": "panel_topology",
    "description": "Panels to topology.",
    "source_kind": "panel_photo",
    "split": "synthetic_clean",
    "prompt": "single_shot",
    "metrics": ["detection_prf1"],
}


def _write_yaml(path: Path, data: Any) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _run_config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "name": "r",
        "task": "task.yaml",
        "catalogue": "models.yaml",
        "models": ["m1"],
    }
    return RunConfig.model_validate({**base, **overrides})


def test_provider_names_match_the_literal():
    assert set(PROVIDER_NAMES) == set(get_args(ProviderName))
    assert len(PROVIDER_NAMES) == len(set(PROVIDER_NAMES)) == 6


def test_enums_values():
    assert [m.value for m in Mode] == ["single_shot", "agentic"]
    assert [t.value for t in TrackerKind] == ["null", "mlflow"]


def test_load_yaml_mapping(tmp_path: Path):
    path = _write_yaml(tmp_path / "m.yaml", {"a": 1, "b": [1, 2]})
    assert load_yaml(path) == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("content", ["- 1\n- 2\n", "just a string\n", "", "42\n"])
def test_load_yaml_non_mapping_raises(tmp_path: Path, content: str):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_yaml(path)


def test_load_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_loaders(tmp_path: Path):
    catalogue = load_model_catalogue(_write_yaml(tmp_path / "models.yaml", CATALOGUE))
    assert catalogue.names() == ["m1", "m2"]
    assert catalogue.get("m2").params == {"max_tokens": 2048}
    assert catalogue.get("m1") == ModelConfig(name="m1", provider="ollama", model_id="qwen2.5vl:7b")

    task = load_task_config(_write_yaml(tmp_path / "task.yaml", TASK))
    assert task == TaskConfig(
        name="panel_topology",
        description="Panels to topology.",
        source_kind=SourceKind.PANEL_PHOTO,
        split="synthetic_clean",
        metrics=["detection_prf1"],
    )

    run = load_run_config(
        _write_yaml(tmp_path / "run.yaml", {"name": "r", "task": "task.yaml", "models": ["m1"]})
    )
    assert run.task == Path("task.yaml")
    assert run.catalogue == Path("configs/models.yaml")
    assert run.mode is Mode.SINGLE_SHOT
    assert run.tracker is TrackerKind.NULL
    assert run.out_dir == Path("runs")
    assert (run.limit, run.seed, run.max_rounds) == (None, 0, 3)


def test_models_reject_extra_keys_and_bad_values():
    with pytest.raises(ValidationError):
        ModelConfig(name="m", provider="ollama", model_id="x", extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"name": "m", "provider": "nope", "model_id": "x"})
    with pytest.raises(ValidationError):
        _run_config(models=[])
    with pytest.raises(ValidationError):
        _run_config(mode="fast")
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"name": "t", "source_kind": "photo", "split": "s"})


def test_catalogue_get_unknown_lists_known_names():
    catalogue = ModelCatalogue.model_validate(CATALOGUE)
    with pytest.raises(KeyError) as info:
        catalogue.get("nope")
    message = str(info.value)
    assert "unknown model 'nope'" in message
    assert "m1" in message
    assert "m2" in message


def test_apply_overrides_changes_only_given_fields():
    original = _run_config()
    updated = apply_overrides(original, models=["m2"], limit=3, out_dir=Path("elsewhere"))
    assert updated.models == ["m2"]
    assert updated.limit == 3
    assert updated.out_dir == Path("elsewhere")
    assert updated.mode is original.mode
    assert updated.seed == original.seed
    assert updated.task == original.task
    # the input is untouched and a no-op override is an equal copy
    assert original.models == ["m1"]
    assert apply_overrides(original) == original
    assert apply_overrides(original, mode=Mode.AGENTIC, seed=9).mode is Mode.AGENTIC


def test_resolve_run_relative_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_yaml(tmp_path / "task.yaml", TASK)
    _write_yaml(tmp_path / "models.yaml", CATALOGUE)
    monkeypatch.chdir(tmp_path)

    config = _run_config(models=["m1", "m2"])
    resolved, task, catalogue = resolve_run(config)
    assert resolved == config
    assert task.name == "panel_topology"
    assert catalogue.names() == ["m1", "m2"]

    with pytest.raises(KeyError, match="unknown model 'ghost'"):
        resolve_run(_run_config(models=["m1", "ghost"]))
    with pytest.raises(FileNotFoundError):
        resolve_run(_run_config(task="missing.yaml"))
    with pytest.raises(ValidationError):
        resolve_run(_run_config(catalogue="task.yaml"))


def test_resolve_run_absolute_paths(tmp_path: Path):
    task = _write_yaml(tmp_path / "task.yaml", TASK)
    catalogue = _write_yaml(tmp_path / "models.yaml", CATALOGUE)
    _, loaded_task, loaded_catalogue = resolve_run(
        _run_config(task=str(task), catalogue=str(catalogue))
    )
    assert loaded_task.split == "synthetic_clean"
    assert loaded_catalogue.get("m2").supports_tools is True


def test_run_layout_and_run_dir():
    assert set(RUN_LAYOUT) == {"config", "predictions", "metrics", "summary", "report"}
    assert len(set(RUN_LAYOUT.values())) == len(RUN_LAYOUT)
    assert RUN_LAYOUT["config"] == "run.yaml"
    assert run_dir(Path("runs"), "2026-09-22-abc") == Path("runs") / "2026-09-22-abc"
