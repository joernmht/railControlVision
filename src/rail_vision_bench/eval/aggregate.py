"""Aggregation of per-scene metrics into the run's metrics table (stub)."""

from __future__ import annotations

from pathlib import Path


def aggregate_run(run_dir: Path) -> Path:
    """Aggregate the per-scene metric rows of a run into ``metrics.parquet``.

    Intended implementation: read ``predictions.jsonl`` and the per-scene
    metric rows, build a ``pandas`` DataFrame keyed by (model, mode, scene,
    metric), aggregate per (model, mode, metric) and write
    ``run_dir / RUN_LAYOUT["metrics"]`` with ``pyarrow``.

    Args:
        run_dir: The run directory.

    Returns:
        The written metrics file.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.aggregate.aggregate_run is not implemented in the skeleton: "
        "aggregate per-scene metrics into metrics.parquet with pandas/pyarrow"
    )
