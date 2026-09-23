"""Run orchestration: the run-directory contract and the three pipeline stages.

A run lives in ``<out_dir>/<run_id>/`` and contains exactly the files named in
:data:`RUN_LAYOUT`; ``bench run`` (:func:`run_benchmark`), ``bench eval``
(:func:`evaluate_run`) and ``bench report`` (:func:`build_report`) are the
three stages that produce them. Every path inside a run config or a manifest
is resolved relative to the current working directory, as in
:func:`rail_vision_bench.config.resolve_run`.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import orjson
import yaml

from rail_vision_bench import SCHEMA_VERSION, __version__
from rail_vision_bench.config import (
    Mode,
    ModelConfig,
    RunConfig,
    TaskConfig,
    load_run_config,
    load_task_config,
    resolve_run,
)
from rail_vision_bench.dataset.manifest import ManifestRow, read_manifest
from rail_vision_bench.eval.records import PredictionRecord
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider
from rail_vision_bench.schema.models import SceneAnnotation, Source
from rail_vision_bench.settings import Settings, get_settings

RUN_LAYOUT: Final[dict[str, str]] = {
    "config": "run.yaml",
    "predictions": "predictions.jsonl",
    "metrics": "metrics.parquet",
    "summary": "summary.json",
    "report": "report.html",
}
"""File names inside a run directory: the resolved config copy, one PredictionRecord
per line, the metric rows, the aggregated summary and the rendered report."""

ProviderFactory = Callable[[ModelConfig, Settings], VisionProvider]
"""Builds the backend of one catalogue entry; tests inject fakes through it."""

_MEDIA_TYPES: Final[dict[str, str]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def run_dir(runs_root: Path, run_id: str) -> Path:
    """Return the directory of one run.

    Args:
        runs_root: The root under which runs are written (``RunConfig.out_dir``).
        run_id: The run identifier.

    Returns:
        ``runs_root / run_id``.
    """
    return runs_root / run_id


def new_run_id(name: str, *, now: datetime | None = None) -> str:
    """Build a run id from the run name and a UTC timestamp.

    Args:
        name: ``RunConfig.name``.
        now: The timestamp; the current time when omitted.

    Returns:
        ``<name>-<YYYYmmddTHHMMSSZ>``.
    """
    stamp = (now if now is not None else datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{stamp}"


def select_scenes(
    rows: Sequence[ManifestRow], *, split: str, seed: int, limit: int | None
) -> list[ManifestRow]:
    """Pick the scenes of a run: the split's rows in a seeded order, cut to ``limit``.

    Args:
        rows: Every manifest row.
        split: ``TaskConfig.split``.
        seed: ``RunConfig.seed``; fixes the order and therefore the ``limit`` subset.
        limit: Keep at most this many scenes; all when None.

    Returns:
        The selected rows.

    Raises:
        ValueError: If the split has no rows or a scene id repeats.
    """
    selected = sorted((row for row in rows if row.split == split), key=lambda row: row.scene_id)
    if not selected:
        msg = f"the manifest has no rows of split {split!r}"
        raise ValueError(msg)
    ids = [row.scene_id for row in selected]
    if len(set(ids)) != len(ids):
        msg = f"split {split!r} repeats a scene_id"
        raise ValueError(msg)
    random.Random(seed).shuffle(selected)
    return selected if limit is None else selected[:limit]


def price(model: ModelConfig, usage: Usage) -> float | None:
    """Return the cost of one completion from the catalogue prices.

    Args:
        model: The catalogue entry (USD per 1k input/output tokens).
        usage: The token counts; a provider-reported ``cost_usd`` wins.

    Returns:
        The cost in USD, or None when the catalogue has no prices.
    """
    if usage.cost_usd is not None:
        return usage.cost_usd
    if model.cost_per_1k_in is None or model.cost_per_1k_out is None:
        return None
    return (
        usage.input_tokens * model.cost_per_1k_in + usage.output_tokens * model.cost_per_1k_out
    ) / 1000


def _default_factory(model: ModelConfig, settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.registry import get_provider

    return get_provider(model.provider, settings)


def _image_input(path: Path) -> ImageInput:
    """Read a scene image; formats other than PNG/JPEG/WebP are re-encoded as PNG."""
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is not None:
        return ImageInput(data=path.read_bytes(), media_type=media_type)  # type: ignore[arg-type]
    import io

    from rail_vision_bench.ingest.image import load_image

    buffer = io.BytesIO()
    load_image(path).save(buffer, format="PNG")
    return ImageInput(data=buffer.getvalue(), media_type="image/png")


async def _attempt(
    provider: VisionProvider,
    row: ManifestRow,
    *,
    model: ModelConfig,
    config: RunConfig,
    run_id: str,
    prompt: str | None,
) -> PredictionRecord:
    """Drive one scene through one model and record the outcome, whatever it is."""
    from rail_vision_bench.agents.graph import run_agentic
    from rail_vision_bench.agents.single_shot import single_shot_attempt
    from rail_vision_bench.graph.validator import validate_document

    base: dict[str, Any] = {
        "scene_id": row.scene_id,
        "run_id": run_id,
        "model": model.name,
        "mode": config.mode,
    }
    started = time.perf_counter()
    try:
        image = _image_input(Path.cwd() / row.image)
        source = Source(kind=row.source_kind, image=row.image, width=row.width, height=row.height)
        if config.mode is Mode.AGENTIC:
            result: Any = await run_agentic(
                provider,
                image,
                model=model.model_id,
                scene_id=row.scene_id,
                max_rounds=config.max_rounds,
                source=source,
            )
        else:
            result = await single_shot_attempt(
                provider,
                image,
                model=model.model_id,
                prompt=prompt,
                scene_id=row.scene_id,
                source=source,
            )
    except Exception as exc:  # a failed call is a data point, not a crashed run
        return PredictionRecord(
            **base,
            latency_ms=(time.perf_counter() - started) * 1000,
            parse_error=f"{type(exc).__name__}: {exc}",
        )
    document = result.annotation if result.annotation is not None else result.document
    return PredictionRecord(
        **base,
        latency_ms=result.latency_ms,
        tokens_in=result.usage.input_tokens,
        tokens_out=result.usage.output_tokens,
        cost_usd=price(model, result.usage),
        raw_text=result.raw_text,
        annotation=result.annotation,
        parse_error=result.parse_error,
        validation=validate_document(document) if document is not None else None,
    )


async def _drive(
    rows: Sequence[ManifestRow],
    models: Sequence[ModelConfig],
    *,
    config: RunConfig,
    run_id: str,
    prompt: str | None,
    providers: Mapping[str, VisionProvider],
    out: Path,
) -> list[PredictionRecord]:
    """Run every (model, scene) pair in order, appending each record as it lands."""
    records: list[PredictionRecord] = []
    with out.open("ab") as handle:
        for model in models:
            for row in rows:
                record = await _attempt(
                    providers[model.name],
                    row,
                    model=model,
                    config=config,
                    run_id=run_id,
                    prompt=prompt,
                )
                handle.write(orjson.dumps(record.model_dump(mode="json")) + b"\n")
                handle.flush()
                records.append(record)
    return records


def _run_counts(records: Sequence[PredictionRecord]) -> dict[str, float]:
    """Headline counters of a finished run for the tracker."""
    parsed = sum(record.annotation is not None for record in records)
    valid = sum(record.validation is not None and record.validation.ok for record in records)
    costs = [record.cost_usd for record in records if record.cost_usd is not None]
    return {
        "predictions": float(len(records)),
        "parsed": float(parsed),
        "parse_failures": float(len(records) - parsed),
        "valid": float(valid),
        "cost_usd_total": float(sum(costs)),
    }


def run_benchmark(
    config: RunConfig,
    *,
    settings: Settings | None = None,
    provider_factory: ProviderFactory | None = None,
    run_id: str | None = None,
) -> Path:
    """Execute a benchmark run and return its run directory.

    Resolves the config (task, catalogue, model names), selects the task
    split's scenes from ``config.manifest`` in a seeded order, writes
    ``RUN_LAYOUT["config"]``, and drives every scene through every model with
    the single-shot baseline (prompt ``task.prompt``) or the agentic loop. Each
    attempt is validated and appended to ``predictions.jsonl`` as soon as it
    finishes; a provider error is recorded as the attempt's ``parse_error``
    rather than aborting the run. The configured tracker receives the params,
    the headline counters and the predictions file.

    Args:
        config: The run to execute.
        settings: Credentials and knobs; ``get_settings()`` when omitted.
        provider_factory: Builds each model's backend; the provider registry when omitted.
        run_id: The run id; :func:`new_run_id` when omitted.

    Returns:
        The run directory.

    Raises:
        FileNotFoundError: If the task, catalogue or manifest file is missing.
        FileExistsError: If the run directory already exists.
        KeyError: If a model name is not in the catalogue.
        ValueError: If a config file is invalid or the split has no scenes.
    """
    from rail_vision_bench.agents.prompts import load_prompt
    from rail_vision_bench.tracking.base import get_tracker

    settings = settings if settings is not None else get_settings()
    factory = provider_factory if provider_factory is not None else _default_factory
    config, task, catalogue = resolve_run(config)
    rows = select_scenes(
        read_manifest(Path.cwd() / config.manifest),
        split=task.split,
        seed=config.seed,
        limit=config.limit,
    )
    models = [catalogue.get(name) for name in config.models]
    prompt = load_prompt(task.prompt) if config.mode is Mode.SINGLE_SHOT else None
    providers = {model.name: factory(model, settings) for model in models}

    run_id = run_id if run_id is not None else new_run_id(config.name)
    out = run_dir(config.out_dir, run_id)
    out.mkdir(parents=True)
    (out / RUN_LAYOUT["config"]).write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    predictions = out / RUN_LAYOUT["predictions"]
    predictions.touch()

    tracker = get_tracker(config.tracker, settings)
    tracker.start_run(
        run_id,
        {
            **config.model_dump(mode="json"),
            "task_split": task.split,
            "task_prompt": task.prompt,
            "model_ids": {model.name: model.model_id for model in models},
            "scenes": len(rows),
        },
    )
    try:
        records = asyncio.run(
            _drive(
                rows,
                models,
                config=config,
                run_id=run_id,
                prompt=prompt,
                providers=providers,
                out=predictions,
            )
        )
        tracker.log_metrics(_run_counts(records))
        tracker.log_artifact(out / RUN_LAYOUT["config"])
        tracker.log_artifact(predictions)
    finally:
        tracker.end_run()
    return out


def load_ground_truth(rows: Sequence[ManifestRow]) -> dict[str, SceneAnnotation]:
    """Load the ground-truth document of every manifest row.

    Args:
        rows: Manifest rows; ``gt`` paths are relative to the working directory.

    Returns:
        The documents keyed by scene id.

    Raises:
        FileNotFoundError: If a ground-truth file is missing.
        ValueError: If a document does not parse or its scene id differs from the row's.
    """
    ground_truth: dict[str, SceneAnnotation] = {}
    for row in rows:
        doc = SceneAnnotation.model_validate_json((Path.cwd() / row.gt).read_bytes())
        if doc.scene_id != row.scene_id:
            msg = (
                f"{row.gt}: scene_id {doc.scene_id!r} differs from the manifest's {row.scene_id!r}"
            )
            raise ValueError(msg)
        ground_truth[row.scene_id] = doc
    return ground_truth


def git_sha() -> str | None:
    """Return the commit of the working directory's checkout, or None outside git."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _load_run(run_path: Path) -> tuple[RunConfig, TaskConfig]:
    config_file = run_path / RUN_LAYOUT["config"]
    if not config_file.is_file():
        msg = f"{run_path} is not a run directory: {RUN_LAYOUT['config']} is missing"
        raise FileNotFoundError(msg)
    config = load_run_config(config_file)
    return config, load_task_config(Path.cwd() / config.task)


def evaluate_run(run_dir: Path) -> Path:
    """Score the predictions of a finished run.

    Reads the run's ``run.yaml`` (task and manifest are resolved relative to
    the working directory), loads the ground truth of every predicted scene,
    scores the task's metrics (all of ``METRIC_NAMES`` when the task lists
    none) into ``RUN_LAYOUT["metrics"]`` and writes ``RUN_LAYOUT["summary"]``:
    run identity, reproducibility inputs (package and schema version, git SHA,
    seed, model ids) and the run-level value of every metric per model and mode.

    Args:
        run_dir: The run directory produced by :func:`run_benchmark`.

    Returns:
        The path of the metrics file.

    Raises:
        FileNotFoundError: If the run config, task, manifest, predictions or a
            ground-truth file is missing.
        ValueError: If a file is invalid.
    """
    from rail_vision_bench.eval.aggregate import aggregate_run, load_metrics, read_predictions

    config, task = _load_run(run_dir)
    predictions_file = run_dir / RUN_LAYOUT["predictions"]
    if not predictions_file.is_file():
        msg = f"{predictions_file} is missing"
        raise FileNotFoundError(msg)
    records = read_predictions(predictions_file)
    predicted = {record.scene_id for record in records}
    rows = [
        row
        for row in read_manifest(Path.cwd() / config.manifest)
        if row.split == task.split and row.scene_id in predicted
    ]
    metrics_path = aggregate_run(
        run_dir, ground_truth=load_ground_truth(rows), metrics=task.metrics or None
    )

    frame = load_metrics(metrics_path, parse_extra=False)
    headline: dict[str, dict[str, float]] = {}
    for item in frame[frame["scope"] == "run"].to_dict("records"):
        headline.setdefault(f"{item['model']}/{item['mode']}", {})[str(item["name"])] = float(
            item["value"]
        )
    summary = {
        "run_id": run_dir.name,
        "name": config.name,
        "task": task.name,
        "split": task.split,
        "mode": config.mode.value,
        "models": config.models,
        "seed": config.seed,
        "limit": config.limit,
        "max_rounds": config.max_rounds,
        "scenes": len(rows),
        "predictions": len(records),
        "parse_failures": sum(record.annotation is None for record in records),
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": headline,
    }
    (run_dir / RUN_LAYOUT["summary"]).write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return metrics_path


def build_report(run_dirs: Sequence[Path], out: Path) -> Path:
    """Render a leaderboard comparing one or more evaluated runs.

    Loads each run's ``summary.json`` and the run-level rows of its
    ``metrics.parquet``, ranks the entries per metric and writes the
    leaderboard to ``out`` (Markdown when ``out`` ends in ``.md``, else HTML).
    Every run directory also receives its own ``RUN_LAYOUT["report"]``.

    Args:
        run_dirs: Evaluated run directories.
        out: Destination of the rendered report.

    Returns:
        The written report path.

    Raises:
        FileNotFoundError: If a run has not been evaluated.
        ValueError: If ``run_dirs`` is empty.
    """
    from rail_vision_bench.eval.aggregate import load_metrics
    from rail_vision_bench.report.render import render_leaderboard

    if not run_dirs:
        msg = "no run directories given"
        raise ValueError(msg)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in run_dirs:
        summary_file = path / RUN_LAYOUT["summary"]
        metrics_file = path / RUN_LAYOUT["metrics"]
        if not summary_file.is_file() or not metrics_file.is_file():
            msg = f"{path} has not been evaluated; run `bench eval {path}` first"
            raise FileNotFoundError(msg)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        frame = load_metrics(metrics_file, parse_extra=True)
        rows = [
            {**row, "run_id": summary.get("run_id", path.name), "split": summary.get("split")}
            for row in frame[frame["scope"] == "run"].to_dict("records")
        ]
        render_leaderboard(rows, path / RUN_LAYOUT["report"], runs=[summary])
        all_rows.extend(rows)
        summaries.append(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    return render_leaderboard(
        all_rows, out, runs=summaries, fmt="md" if out.suffix.lower() == ".md" else "html"
    )
