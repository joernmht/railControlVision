"""The three pipeline stages end to end: run -> eval -> report on a tiny synthetic split.

Paths inside run configs and manifests are relative to the working directory,
so every test runs from ``tmp_path``. The provider is a fake that answers with
the scene's ground truth (or with junk, or an exception, for chosen scenes), so
the metric values are known exactly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from rail_vision_bench.cli import EXIT_INVALID, EXIT_OK, app
from rail_vision_bench.config import Mode, ModelConfig, RunConfig, TrackerKind
from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest
from rail_vision_bench.eval.aggregate import load_metrics, read_predictions
from rail_vision_bench.providers.base import Usage, VisionProvider, VisionRequest, VisionResponse
from rail_vision_bench.runner import (
    RUN_LAYOUT,
    build_report,
    evaluate_run,
    new_run_id,
    price,
    run_benchmark,
    select_scenes,
)
from rail_vision_bench.schema.models import SourceKind
from rail_vision_bench.settings import Settings
from rail_vision_bench.synth.generate import generate_dataset


class OracleProvider:
    """Answers every single-shot request with the ground truth of the scene it names."""

    name: ClassVar[str] = "oracle"

    def __init__(self, truth: dict[str, dict[str, Any]], *, junk: str = "", boom: str = ""):
        self.truth = truth
        self.junk = junk
        self.boom = boom
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        # the longest id wins so that "s-1" never shadows "s-10"
        scene = max((sid for sid in self.truth if sid in request.prompt), key=len)
        if scene == self.boom:
            msg = "provider exploded"
            raise RuntimeError(msg)
        usage = Usage(input_tokens=1000, output_tokens=500)
        if scene == self.junk:
            return VisionResponse(text="I cannot read this panel.", usage=usage, latency_ms=5.0)
        doc = self.truth[scene]
        return VisionResponse(text=json.dumps(doc), parsed=doc, usage=usage, latency_ms=10.0)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A synthetic split, a task, a two-model catalogue and a run config under tmp_path."""
    monkeypatch.chdir(tmp_path)
    manifest = generate_dataset(Path("data/synthetic"), n=4, seed=5)
    rows = read_manifest(manifest)
    truth = {row.scene_id: json.loads(Path(row.gt).read_text(encoding="utf-8")) for row in rows}
    Path("task.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "panel_topology",
                "source_kind": "synthetic",
                "split": "synthetic_clean",
                "prompt": "single_shot",
                "metrics": ["detection_prf1", "state_accuracy", "latency_p50_ms", "cost_usd"],
            }
        ),
        encoding="utf-8",
    )
    Path("models.yaml").write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "oracle",
                        "provider": "ollama",
                        "model_id": "oracle-1",
                        "cost_per_1k_in": 0.001,
                        "cost_per_1k_out": 0.002,
                    },
                    {"name": "oracle-free", "provider": "ollama", "model_id": "oracle-2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = RunConfig(
        name="e2e",
        task=Path("task.yaml"),
        catalogue=Path("models.yaml"),
        manifest=manifest,
        models=["oracle"],
        out_dir=Path("runs"),
    )
    return {"rows": rows, "truth": truth, "config": config}


def _factory(provider: VisionProvider) -> Any:
    def build(model: ModelConfig, settings: Settings) -> VisionProvider:
        return provider

    return build


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_run_eval_report_with_a_perfect_model(workspace: dict[str, Any]):
    oracle = OracleProvider(workspace["truth"])
    out = run_benchmark(
        workspace["config"], settings=_settings(), provider_factory=_factory(oracle), run_id="r1"
    )
    assert out == Path("runs") / "r1"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [RUN_LAYOUT["config"], RUN_LAYOUT["predictions"]]
    )
    stored = yaml.safe_load((out / RUN_LAYOUT["config"]).read_text(encoding="utf-8"))
    assert stored["models"] == ["oracle"]

    records = read_predictions(out / RUN_LAYOUT["predictions"])
    assert len(records) == 4
    assert {r.scene_id for r in records} == {row.scene_id for row in workspace["rows"]}
    for record in records:
        assert record.annotation is not None
        assert record.parse_error is None
        assert record.validation is not None
        assert record.validation.ok
        assert record.tokens_in == 1000
        assert record.cost_usd == pytest.approx(1000 * 0.001 / 1000 + 500 * 0.002 / 1000)
    assert all(request.model == "oracle-1" for request in oracle.requests)

    metrics_path = evaluate_run(out)
    assert metrics_path == out / RUN_LAYOUT["metrics"]
    frame = load_metrics(metrics_path)
    run_rows = frame[frame["scope"] == "run"].set_index("name")
    assert run_rows.loc["detection_prf1", "value"] == pytest.approx(1.0)
    assert run_rows.loc["state_accuracy", "value"] == pytest.approx(1.0)
    assert run_rows.loc["cost_usd", "value"] == pytest.approx(4 * 0.002)

    summary = json.loads((out / RUN_LAYOUT["summary"]).read_text(encoding="utf-8"))
    assert summary["run_id"] == "r1"
    assert summary["split"] == "synthetic_clean"
    assert summary["predictions"] == 4
    assert summary["parse_failures"] == 0
    assert summary["schema_version"] == "v0"
    assert summary["metrics"]["oracle/single_shot"]["detection_prf1"] == pytest.approx(1.0)

    board = build_report([out], Path("reports/board.html"))
    assert board.is_file()
    html = board.read_text(encoding="utf-8")
    assert "oracle" in html
    assert "detection_prf1" in html
    assert (out / RUN_LAYOUT["report"]).is_file()


def test_failures_are_recorded_not_raised(workspace: dict[str, Any]):
    ids = sorted(workspace["truth"])
    oracle = OracleProvider(workspace["truth"], junk=ids[0], boom=ids[1])
    out = run_benchmark(
        workspace["config"], settings=_settings(), provider_factory=_factory(oracle), run_id="r2"
    )
    by_scene = {r.scene_id: r for r in read_predictions(out / RUN_LAYOUT["predictions"])}
    junk, boom = by_scene[ids[0]], by_scene[ids[1]]
    assert junk.annotation is None
    assert junk.raw_text == "I cannot read this panel."
    assert junk.parse_error
    assert boom.annotation is None
    assert boom.parse_error == "RuntimeError: provider exploded"
    assert boom.tokens_in is None

    evaluate_run(out)
    frame = load_metrics(out / RUN_LAYOUT["metrics"])
    detection = frame[(frame["scope"] == "run") & (frame["name"] == "detection_prf1")]
    # two scenes are perfect, two contribute only false negatives
    assert 0.0 < float(detection["value"].iloc[0]) < 1.0
    summary = json.loads((out / RUN_LAYOUT["summary"]).read_text(encoding="utf-8"))
    assert summary["parse_failures"] == 2


def test_limit_seed_and_several_models(workspace: dict[str, Any]):
    oracle = OracleProvider(workspace["truth"])
    config = workspace["config"].model_copy(
        update={"limit": 2, "seed": 3, "models": ["oracle", "oracle-free"]}
    )
    out = run_benchmark(
        config, settings=_settings(), provider_factory=_factory(oracle), run_id="r3"
    )
    records = read_predictions(out / RUN_LAYOUT["predictions"])
    assert len(records) == 4  # 2 scenes x 2 models
    free = [r for r in records if r.model == "oracle-free"]
    assert all(r.cost_usd is None for r in free)
    expected = [
        row.scene_id
        for row in select_scenes(workspace["rows"], split="synthetic_clean", seed=3, limit=2)
    ]
    assert [r.scene_id for r in records if r.model == "oracle"] == expected


def test_existing_run_directory_is_refused(workspace: dict[str, Any]):
    oracle = OracleProvider(workspace["truth"])
    run_benchmark(
        workspace["config"], settings=_settings(), provider_factory=_factory(oracle), run_id="dup"
    )
    with pytest.raises(FileExistsError):
        run_benchmark(
            workspace["config"],
            settings=_settings(),
            provider_factory=_factory(oracle),
            run_id="dup",
        )


def test_agentic_mode_records_every_scene(workspace: dict[str, Any]):
    class Silent:
        name: ClassVar[str] = "silent"

        async def complete(self, request: VisionRequest) -> VisionResponse:
            return VisionResponse(text="{}")

    config = workspace["config"].model_copy(update={"mode": Mode.AGENTIC, "max_rounds": 1})
    out = run_benchmark(
        config, settings=_settings(), provider_factory=_factory(Silent()), run_id="ag"
    )
    records = read_predictions(out / RUN_LAYOUT["predictions"])
    assert len(records) == 4
    assert all(r.mode is Mode.AGENTIC for r in records)


def test_missing_manifest(workspace: dict[str, Any]):
    config = workspace["config"].model_copy(update={"manifest": Path("nope.jsonl")})
    with pytest.raises(FileNotFoundError):
        run_benchmark(config, settings=_settings(), provider_factory=_factory(OracleProvider({})))
    assert not Path("runs").exists()


def test_cli_pipeline(workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    oracle = OracleProvider(workspace["truth"])
    monkeypatch.setattr("rail_vision_bench.runner._default_factory", _factory(oracle))
    config = workspace["config"].model_copy(update={"tracker": TrackerKind.NULL})
    Path("run.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    cli = CliRunner()
    result = cli.invoke(app, ["run", "--config", "run.yaml", "--limit", "2"])
    assert result.exit_code == EXIT_OK, result.output
    (run_path,) = Path("runs").iterdir()
    assert run_path.name.startswith("e2e-")

    result = cli.invoke(app, ["eval", str(run_path)])
    assert result.exit_code == EXIT_OK, result.output
    result = cli.invoke(app, ["report", str(run_path), "--out", "board.md"])
    assert result.exit_code == EXIT_OK, result.output
    assert "oracle" in Path("board.md").read_text(encoding="utf-8")


def test_cli_errors(workspace: dict[str, Any], tmp_path: Path):
    cli = CliRunner()
    empty = tmp_path / "empty"
    empty.mkdir()
    result = cli.invoke(app, ["eval", str(empty)])
    assert result.exit_code == EXIT_INVALID
    assert "run.yaml" in result.output
    result = cli.invoke(app, ["report", str(empty), "--out", "x.html"])
    assert result.exit_code == EXIT_INVALID
    assert "bench eval" in result.output
    assert not Path("x.html").exists()


def _row(scene_id: str, split: str = "synthetic_clean") -> ManifestRow:
    return ManifestRow(
        scene_id=scene_id,
        split=split,
        partition="dev",
        source_kind=SourceKind.SYNTHETIC,
        image=f"{scene_id}.png",
        gt=f"{scene_id}.json",
        width=10,
        height=10,
    )


def test_select_scenes_is_seeded_and_filters_the_split():
    rows = [_row(f"s{i}") for i in range(10)] + [_row("other", split="panel_photo")]
    first = select_scenes(rows, split="synthetic_clean", seed=1, limit=None)
    assert sorted(r.scene_id for r in first) == sorted(f"s{i}" for i in range(10))
    # the order depends on the seed, not on the manifest order
    assert select_scenes(list(reversed(rows)), split="synthetic_clean", seed=1, limit=None) == first
    assert select_scenes(rows, split="synthetic_clean", seed=1, limit=3) == first[:3]
    with pytest.raises(ValueError, match="no rows"):
        select_scenes(rows, split="estw_screen", seed=0, limit=None)
    with pytest.raises(ValueError, match="repeats"):
        select_scenes([_row("a"), _row("a")], split="synthetic_clean", seed=0, limit=None)


def test_price_and_run_id():
    model = ModelConfig(
        name="m", provider="openai", model_id="x", cost_per_1k_in=0.5, cost_per_1k_out=1.0
    )
    assert price(model, Usage(input_tokens=2000, output_tokens=1000)) == pytest.approx(2.0)
    assert price(model, Usage(input_tokens=1, cost_usd=0.25)) == 0.25
    unpriced = ModelConfig(name="m", provider="openai", model_id="x")
    assert price(unpriced, Usage(input_tokens=10)) is None
    run_id = new_run_id("smoke", now=datetime(2026, 9, 23, 10, 11, 12, tzinfo=UTC))
    assert run_id == "smoke-20260923T101112Z"
