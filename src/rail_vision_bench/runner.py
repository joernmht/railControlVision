"""Run orchestration: the run-directory contract (real) and the pipeline stubs.

A run lives in ``<out_dir>/<run_id>/`` and contains exactly the files named in
:data:`RUN_LAYOUT`; ``bench run``, ``bench eval`` and ``bench report`` are the
three stages that produce them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from rail_vision_bench.config import RunConfig
from rail_vision_bench.settings import Settings

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


def run_benchmark(config: RunConfig, *, settings: Settings | None = None) -> Path:
    """Execute a benchmark run and return its run directory.

    Intended orchestration: resolve the config (task, catalogue, model names),
    seed the random sources, create the run directory and write
    ``RUN_LAYOUT["config"]``, obtain a provider per model from the registry,
    drive each scene of the task split (from the dataset manifest) through the
    single-shot baseline or the agentic loop, validate every prediction, append
    one ``PredictionRecord`` per attempt to ``predictions.jsonl`` and log the
    run through the configured tracker.

    Args:
        config: The run to execute.
        settings: Credentials and knobs; ``get_settings()`` when omitted.

    Returns:
        The run directory.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.runner.run_benchmark is not implemented in the skeleton: "
        "drive every scene of the task through the selected models and write predictions.jsonl"
    )


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
