"""Run orchestration: the run-directory contract (real) and the pipeline stubs.

A run lives in ``<out_dir>/<run_id>/`` and contains exactly the files named in
:data:`RUN_LAYOUT`; ``bench run``, ``bench eval`` and ``bench report`` are the
three stages that produce them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
import orjson
import yaml
from pydantic import SecretStr

from rail_vision_bench.config import Mode, ModelConfig, RunConfig, resolve_run
from rail_vision_bench.eval.records import PredictionRecord
from rail_vision_bench.providers.base import Usage, VisionProvider
from rail_vision_bench.settings import Settings

if TYPE_CHECKING:
    from rail_vision_bench.agents.single_shot import SceneInput

logger = logging.getLogger(__name__)

RETRY_WAITS_S: Final[tuple[float, ...]] = (15.0, 60.0)
"""Pauses before the second and third try of a transiently failing request."""

TRANSIENT_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

RUN_LAYOUT: Final[dict[str, str]] = {
    "config": "run.yaml",
    "predictions": "predictions.jsonl",
    "metrics": "metrics.parquet",
    "summary": "summary.json",
    "report": "report.html",
}
"""File names inside a run directory: the resolved config copy, one PredictionRecord
per line, the metric rows, the aggregated summary and the rendered report."""


def run_dir(runs_root: Path, run_id: str) -> Path:
    """Return the directory of one run.

    Args:
        runs_root: The root under which runs are written (``RunConfig.out_dir``).
        run_id: The run identifier.

    Returns:
        ``runs_root / run_id``.
    """
    return runs_root / run_id


def _model_settings(settings: Settings, model: ModelConfig) -> Settings:
    """Apply a catalogue entry's ``base_url`` / ``api_key_env`` params to the settings.

    Raises:
        KeyError: If ``api_key_env`` names an unset environment variable.
    """
    update: dict[str, Any] = {}
    if model.params.get("base_url"):
        update["openai_base_url"] = str(model.params["base_url"])
    key_env = model.params.get("api_key_env")
    if key_env:
        if not os.environ.get(str(key_env)):
            msg = f"model {model.name!r} needs the environment variable {key_env}"
            raise KeyError(msg)
        update["openai_api_key"] = SecretStr(os.environ[str(key_env)])
    return settings.model_copy(update=update)


def _cost(model: ModelConfig, usage: Usage) -> float | None:
    """The billed cost when the provider reports it, else the catalogue estimate."""
    if usage.cost_usd is not None:
        return usage.cost_usd
    if model.cost_per_1k_in is None or model.cost_per_1k_out is None:
        return None
    return (
        usage.input_tokens * model.cost_per_1k_in + usage.output_tokens * model.cost_per_1k_out
    ) / 1000.0


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Read a ``predictions.jsonl``; a missing file is an empty list."""
    if not path.is_file():
        return []
    return [
        PredictionRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


async def _attempt(
    provider: VisionProvider,
    model: ModelConfig,
    scene: SceneInput,
    run_id: str,
    attempt: int,
    prompt: str,
) -> PredictionRecord:
    """One scene through one model, with retries on transient provider errors."""
    from rail_vision_bench.agents.single_shot import run_single_shot
    from rail_vision_bench.graph.validator import validate_scene
    from rail_vision_bench.providers.openai_compat import ProviderError

    last_error = ""
    for retry, wait_s in enumerate((0.0, *RETRY_WAITS_S)):
        if wait_s:
            await asyncio.sleep(wait_s)
        try:
            result = await run_single_shot(
                provider,
                model.model_id,
                scene,
                run_id=run_id,
                prompt=prompt,
                extra=dict(model.params.get("extra_body") or {}),
            )
        except ProviderError as exc:
            last_error = f"provider: {exc}"
            if exc.status_code not in TRANSIENT_STATUS:
                break
        except httpx.HTTPError as exc:
            last_error = f"provider: {type(exc).__name__}: {exc}"
        else:
            usage = result.response.usage
            return PredictionRecord(
                scene_id=scene.scene_id,
                run_id=run_id,
                model=model.name,
                mode=Mode.SINGLE_SHOT,
                attempt=attempt,
                latency_ms=result.response.latency_ms,
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=_cost(model, usage),
                raw_text=result.response.text,
                annotation=result.annotation,
                parse_error=result.parse_error,
                validation=validate_scene(result.annotation) if result.annotation else None,
            )
        logger.warning("%s %s: retry %d after %s", model.name, scene.scene_id, retry, last_error)
    return PredictionRecord(
        scene_id=scene.scene_id,
        run_id=run_id,
        model=model.name,
        mode=Mode.SINGLE_SHOT,
        attempt=attempt,
        latency_ms=0.0,
        parse_error=last_error,
    )


def _run_agentic(config: RunConfig) -> Path:
    """Drive the scenes through the agentic loop.

    Intended orchestration: as the single-shot path, but each scene goes through
    ``agents.graph.build_agent_graph`` for at most ``config.max_rounds`` rounds.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.runner._run_agentic is not implemented in the skeleton: "
        "drive every scene through the LangGraph agent loop"
    )


def run_benchmark(
    config: RunConfig,
    *,
    settings: Settings | None = None,
    retry_failed: bool = False,
    concurrency: int = 3,
) -> Path:
    """Execute a benchmark run and return its run directory.

    Resolves the config, writes ``RUN_LAYOUT["config"]``, and drives every scene of the
    task split (``data/gt/<split>/manifest.jsonl``, filtered to the task's source kind and
    cut to ``config.limit``) through every model in single-shot mode with the task's
    prompt and each model's ``params.extra_body`` (provider-specific request fields), appending one
    ``PredictionRecord`` per attempt to ``predictions.jsonl``. A run is resumable: a
    (scene, model) pair that already has a record is skipped, unless ``retry_failed`` is
    set and its last record has no parsed document, in which case it gets another attempt.

    Args:
        config: The run to execute; ``config.name`` is the run id.
        settings: Credentials and knobs; ``get_settings()`` when omitted.
        retry_failed: Re-attempt pairs whose last attempt produced no document.
        concurrency: Parallel requests per model.

    Returns:
        The run directory.
    """
    from rail_vision_bench.agents.prompts import load_prompt
    from rail_vision_bench.agents.single_shot import SceneInput
    from rail_vision_bench.dataset.manifest import manifest_path, read_manifest
    from rail_vision_bench.providers.registry import get_provider
    from rail_vision_bench.settings import get_settings

    config, task, catalogue = resolve_run(config)
    if config.mode is Mode.AGENTIC:
        return _run_agentic(config)
    settings = settings if settings is not None else get_settings()
    prompt = load_prompt(task.prompt)
    rows = [
        row
        for row in read_manifest(manifest_path(settings.rvb_data_dir, task.split))
        if row.source_kind == task.source_kind
    ][: config.limit]
    out = run_dir(config.out_dir, config.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_LAYOUT["config"]).write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    predictions = out / RUN_LAYOUT["predictions"]
    last: dict[tuple[str, str], PredictionRecord] = {
        (record.scene_id, record.model): record for record in read_predictions(predictions)
    }
    jobs: list[tuple[ModelConfig, SceneInput, int]] = []
    for name in config.models:
        model = catalogue.get(name)
        for row in rows:
            previous = last.get((row.scene_id, name))
            if previous is not None and not (retry_failed and previous.annotation is None):
                continue
            scene = SceneInput(
                scene_id=row.scene_id,
                source_kind=row.source_kind.value,
                image=Path(row.image),
                width=row.width,
                height=row.height,
            )
            jobs.append((model, scene, 1 if previous is None else previous.attempt + 1))
    providers = {
        name: get_provider(
            catalogue.get(name).provider, _model_settings(settings, catalogue.get(name))
        )
        for name in config.models
    }

    async def drive() -> None:
        lock = asyncio.Lock()
        gates = {name: asyncio.Semaphore(concurrency) for name in config.models}

        async def one(model: ModelConfig, scene: SceneInput, attempt: int) -> None:
            async with gates[model.name]:
                record = await _attempt(
                    providers[model.name], model, scene, config.name, attempt, prompt
                )
            async with lock:
                with predictions.open("ab") as handle:
                    handle.write(orjson.dumps(record.model_dump(mode="json")) + b"\n")
            logger.info("%s %s: %s", model.name, scene.scene_id, record.parse_error or "parsed")

        await asyncio.gather(*(one(*job) for job in jobs))

    asyncio.run(drive())
    return out


def evaluate_run(run_dir: Path) -> Path:
    """Score the predictions of a finished run.

    Intended orchestration: read ``predictions.jsonl`` and the ground truth of
    every scene, match elements, compute every metric of the task, and write
    ``RUN_LAYOUT["metrics"]`` plus ``RUN_LAYOUT["summary"]``.

    Args:
        run_dir: The run directory produced by :func:`run_benchmark`.

    Returns:
        The path of the metrics file.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.runner.evaluate_run is not implemented in the skeleton: "
        "score predictions.jsonl against the ground truth into metrics.parquet"
    )


def build_report(run_dirs: Sequence[Path], out: Path) -> Path:
    """Render a leaderboard comparing one or more evaluated runs.

    Intended orchestration: load each run's summary and metrics, rank the
    models per metric, and render the leaderboard template to ``out``.

    Args:
        run_dirs: Evaluated run directories.
        out: Destination of the rendered report.

    Returns:
        The written report path.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.runner.build_report is not implemented in the skeleton: "
        "render the leaderboard of the given runs to a report file"
    )
