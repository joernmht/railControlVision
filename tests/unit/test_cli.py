"""CLI behaviour through typer's CliRunner: exit codes, help, validate, schema export, run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from rail_vision_bench import __version__
from rail_vision_bench.cli import EXIT_INVALID, EXIT_NOT_IMPLEMENTED, EXIT_OK, EXIT_USAGE, app
from tests.conftest import EXAMPLES_DIR, FIXTURES_DIR, REPO_ROOT

HELP_COMMANDS = [
    [],
    ["validate"],
    ["schema", "export"],
    ["run"],
    ["eval"],
    ["report"],
    ["synth", "generate"],
    ["serve"],
]


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _run_setup(tmp_path: Path, *, models: list[str] | None = None) -> Path:
    """Write a task, a catalogue and a run config (absolute paths) and return the run config."""
    task = _write_yaml(
        tmp_path / "task.yaml",
        {"name": "t", "source_kind": "synthetic", "split": "synthetic_clean", "metrics": []},
    )
    catalogue = _write_yaml(
        tmp_path / "models.yaml",
        {"models": [{"name": "m1", "provider": "ollama", "model_id": "qwen2.5vl:7b"}]},
    )
    return _write_yaml(
        tmp_path / "run.yaml",
        {
            "name": "demo-run",
            "task": str(task),
            "catalogue": str(catalogue),
            "models": models if models is not None else ["m1"],
        },
    )


def test_version(cli: CliRunner):
    result = cli.invoke(app, ["--version"])
    assert result.exit_code == EXIT_OK
    assert __version__ in result.output
    assert result.output.startswith("bench ")


@pytest.mark.parametrize("command", HELP_COMMANDS, ids=lambda c: " ".join(c) or "bench")
def test_help(cli: CliRunner, command: list[str]):
    result = cli.invoke(app, [*command, "--help"])
    assert result.exit_code == EXIT_OK
    assert "Usage" in result.output


def test_no_arguments_shows_help(cli: CliRunner):
    result = cli.invoke(app, [])
    assert "Usage" in result.output


def test_unknown_command_is_usage_error(cli: CliRunner):
    result = cli.invoke(app, ["frobnicate"])
    assert result.exit_code == EXIT_USAGE


@pytest.mark.parametrize("name", ["minimal.json", "station_dkw.json"])
def test_validate_examples_strict(cli: CliRunner, name: str):
    result = cli.invoke(app, ["validate", "--strict", str(EXAMPLES_DIR / name)])
    assert result.exit_code == EXIT_OK, result.output
    assert "0 error(s)" in result.output


def test_validate_all_examples_at_once(cli: CliRunner):
    files = sorted(str(p) for p in EXAMPLES_DIR.glob("*.json"))
    result = cli.invoke(app, ["validate", "--strict", *files])
    assert result.exit_code == EXIT_OK, result.output
    assert f"{len(files)} file(s) checked" in result.output


def test_validate_with_schema_file(cli: CliRunner):
    result = cli.invoke(
        app,
        [
            "validate",
            "--schema",
            str(REPO_ROOT / "schema" / "v0.json"),
            str(EXAMPLES_DIR / "minimal.json"),
        ],
    )
    assert result.exit_code == EXIT_OK, result.output


def test_validate_invalid_fixture_exits_1(cli: CliRunner):
    result = cli.invoke(app, ["validate", str(FIXTURES_DIR / "invalid" / "duplicate_id.json")])
    assert result.exit_code == EXIT_INVALID
    assert "ID_DUPLICATE" in result.output


def test_validate_strict_fixture_flips_with_flag(cli: CliRunner):
    fixture = str(FIXTURES_DIR / "strict" / "state_missing.json")
    assert cli.invoke(app, ["validate", fixture]).exit_code == EXIT_OK
    assert cli.invoke(app, ["validate", "--strict", fixture]).exit_code == EXIT_INVALID


def test_validate_json_output(cli: CliRunner):
    fixture = FIXTURES_DIR / "invalid" / "dangling_ref.json"
    result = cli.invoke(app, ["validate", "--json", str(fixture)])
    assert result.exit_code == EXIT_INVALID
    parsed = json.loads(result.output)
    assert list(parsed) == [str(fixture)]
    report = parsed[str(fixture)]
    assert report["ok"] is False
    assert "TOPO_DANGLING_REF" in {issue["code"] for issue in report["issues"]}


def test_validate_json_output_ok(cli: CliRunner):
    example = EXAMPLES_DIR / "minimal.json"
    result = cli.invoke(app, ["validate", "--json", "--strict", str(example)])
    assert result.exit_code == EXIT_OK
    assert json.loads(result.output) == {str(example): {"ok": True, "issues": []}}


@pytest.mark.parametrize(
    ("content", "fragment"),
    [("[1, 2", "not valid JSON"), ("[1, 2]", "expected a JSON object")],
)
def test_validate_rejects_non_object_files(
    cli: CliRunner, tmp_path: Path, content: str, fragment: str
):
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    result = cli.invoke(app, ["validate", "--json", str(bad)])
    assert result.exit_code == EXIT_INVALID
    report = json.loads(result.output)[str(bad)]
    assert report["ok"] is False
    assert [issue["code"] for issue in report["issues"]] == ["SCHEMA_INVALID"]
    assert fragment in report["issues"][0]["message"]


def test_validate_missing_file_is_usage_error(cli: CliRunner, tmp_path: Path):
    result = cli.invoke(app, ["validate", str(tmp_path / "nope.json")])
    assert result.exit_code == EXIT_USAGE


def test_schema_export_then_check_then_drift(cli: CliRunner, tmp_path: Path):
    out = tmp_path / "nested" / "v0.json"
    result = cli.invoke(app, ["schema", "export", "--out", str(out)])
    assert result.exit_code == EXIT_OK, result.output
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == (REPO_ROOT / "schema" / "v0.json").read_text(
        encoding="utf-8"
    )

    result = cli.invoke(app, ["schema", "export", "--out", str(out), "--check"])
    assert result.exit_code == EXIT_OK, result.output

    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = cli.invoke(app, ["schema", "export", "--out", str(out), "--check"])
    assert result.exit_code == EXIT_INVALID
    assert "out of date" in result.output


def test_schema_check_committed_file(cli: CliRunner, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    result = cli.invoke(app, ["schema", "export", "--check"])
    assert result.exit_code == EXIT_OK, result.output


def test_schema_check_missing_file(cli: CliRunner, tmp_path: Path):
    result = cli.invoke(
        app, ["schema", "export", "--out", str(tmp_path / "missing.json"), "--check"]
    )
    assert result.exit_code == EXIT_INVALID
    assert "out of date" in result.output


def test_run_dry_run(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path)
    result = cli.invoke(app, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code == EXIT_OK, result.output
    assert "demo-run" in result.output
    resolved = yaml.safe_load(result.output)
    assert resolved["name"] == "demo-run"
    assert resolved["models"] == ["m1"]
    assert resolved["mode"] == "single_shot"
    assert resolved["tracker"] == "null"


def test_run_dry_run_applies_overrides(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path)
    result = cli.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dry-run",
            "--mode",
            "agentic",
            "--limit",
            "5",
            "--seed",
            "7",
            "--out",
            "elsewhere",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    resolved = yaml.safe_load(result.output)
    assert resolved["mode"] == "agentic"
    assert resolved["limit"] == 5
    assert resolved["seed"] == 7
    assert resolved["out_dir"] == "elsewhere"


def test_run_dry_run_unknown_model(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path, models=["ghost"])
    result = cli.invoke(app, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code == EXIT_INVALID
    assert "unknown model 'ghost'" in result.output
    assert "m1" in result.output


def test_run_dry_run_unknown_model_override(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path)
    result = cli.invoke(app, ["run", "--config", str(config), "--dry-run", "--model", "ghost"])
    assert result.exit_code == EXIT_INVALID
    assert "unknown model 'ghost'" in result.output


def test_run_dry_run_missing_task_file(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path)
    (tmp_path / "task.yaml").unlink()
    result = cli.invoke(app, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code == EXIT_INVALID
    assert "task.yaml" in result.output


def test_run_dry_run_invalid_config(cli: CliRunner, tmp_path: Path):
    config = _write_yaml(tmp_path / "run.yaml", {"name": "x", "task": "t.yaml", "models": []})
    result = cli.invoke(app, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code == EXIT_INVALID


def test_run_without_dry_run_is_not_implemented(cli: CliRunner, tmp_path: Path):
    config = _run_setup(tmp_path)
    result = cli.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "[not implemented]" in result.output
    assert "run_benchmark" in result.output


def test_eval_is_not_implemented(cli: CliRunner, tmp_path: Path):
    result = cli.invoke(app, ["eval", str(tmp_path)])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "evaluate_run" in result.output


def test_report_is_not_implemented(cli: CliRunner, tmp_path: Path):
    result = cli.invoke(app, ["report", str(tmp_path), "--out", str(tmp_path / "lb.html")])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "build_report" in result.output
    assert not (tmp_path / "lb.html").exists()
