"""Peripheral CLI commands: synth generate (stub), serve --help and run without --dry-run.

These commands reach code outside the schema/validator core, so they live apart
from test_cli.py; the configs are written into tmp_path so nothing here depends
on the repository's configs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from rail_vision_bench.cli import EXIT_INVALID, EXIT_NOT_IMPLEMENTED, EXIT_OK, EXIT_USAGE, app


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _run_config(tmp_path: Path) -> Path:
    """A valid task + catalogue + run config whose out_dir lies inside tmp_path."""
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
            "name": "peripheral-run",
            "task": str(task),
            "catalogue": str(catalogue),
            "models": ["m1"],
            "out_dir": str(tmp_path / "runs"),
        },
    )


def test_synth_generate_writes_a_dataset(cli: CliRunner, tmp_path: Path):
    from rail_vision_bench.dataset.manifest import read_manifest

    out = tmp_path / "synthetic"
    result = cli.invoke(app, ["synth", "generate", "--out", str(out), "--n", "2", "--seed", "1"])
    assert result.exit_code == EXIT_OK, result.output
    rows = read_manifest(out / "manifest.jsonl")
    assert len(rows) == 2
    assert all(row.split == "synthetic_clean" for row in rows)
    assert len(list((out / "images").glob("*.png"))) == 2
    assert len(list((out / "gt").glob("*.json"))) == 2


def test_synth_generate_unknown_preset(cli: CliRunner, tmp_path: Path):
    out = tmp_path / "synthetic"
    result = cli.invoke(app, ["synth", "generate", "--out", str(out), "--augment", "ghost"])
    assert result.exit_code == EXIT_INVALID, result.output
    assert "ghost" in result.output


def test_synth_generate_requires_out(cli: CliRunner):
    result = cli.invoke(app, ["synth", "generate"])
    assert result.exit_code == EXIT_USAGE


def test_serve_help(cli: CliRunner):
    result = cli.invoke(app, ["serve", "--help"])
    assert result.exit_code == EXIT_OK, result.output
    for option in ("--host", "--port", "--reload"):
        assert option in result.output


def test_run_without_dry_run_is_not_implemented(cli: CliRunner, tmp_path: Path):
    config = _run_config(tmp_path)
    result = cli.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == EXIT_NOT_IMPLEMENTED, result.output
    assert "[not implemented]" in result.output
    assert "run_benchmark" in result.output
    assert not (tmp_path / "runs").exists(), "the stub must not create the run directory"
