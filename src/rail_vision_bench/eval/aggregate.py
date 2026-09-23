"""Aggregation of per-scene metrics into the run's metrics table.

``metrics.parquet`` has one row per (model, mode, scope, scene_id, name) with
the columns ``model``, ``mode``, ``scope`` (``"run"`` for the aggregate of a
model and mode, ``"scene"`` for a per-scene row), ``scene_id`` (None on run
rows), ``name``, ``value``, ``n`` and ``extra`` (the row's ``extra`` dict as a
JSON string). Run rows of per-scene metrics are micro-averaged from the raw
counts in ``extra`` (:func:`combine_results`); latency and cost rows exist only
at run scope and come from the prediction records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rail_vision_bench.eval.metrics import (
    METRIC_NAMES,
    detection_result,
    ece_result,
    latency_summary,
    prf1_counts,
    ratio,
)
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.eval.scoring import score_prediction
from rail_vision_bench.schema.models import SceneAnnotation

COLUMNS: tuple[str, ...] = ("model", "mode", "scope", "scene_id", "name", "value", "n", "extra")
"""Columns of ``metrics.parquet``, in order."""


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Read a ``predictions.jsonl`` file; blank lines are skipped.

    Args:
        path: The JSON Lines file.

    Returns:
        The records in file order.
    """
    return [
        PredictionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_attempts(records: Sequence[PredictionRecord]) -> list[PredictionRecord]:
    """Keep the highest attempt per (model, mode, scene); later lines win ties.

    Args:
        records: Prediction records, possibly several attempts per scene.

    Returns:
        One record per (model, mode, scene), in first-seen order.
    """
    latest: dict[tuple[str, str, str], PredictionRecord] = {}
    for record in records:
        key = (record.model, str(record.mode), record.scene_id)
        if key not in latest or record.attempt >= latest[key].attempt:
            latest[key] = record
    return list(latest.values())


def combine_results(results: Sequence[MetricResult]) -> MetricResult:
    """Micro-aggregate rows of one metric name, e.g. across the scenes of a run.

    Counts in ``extra`` are summed and the value is recomputed from them, so
    the result equals the metric computed over the pooled items: PRF1 from
    summed ``tp``/``fp``/``fn`` (per family too), error rates from summed
    ``errors``/``ref_len``, state accuracy from summed ``correct``/``total``,
    topology agreement from summed ``intersection``/``union``, Brier from the
    summed squared error and ECE from summed bins. Other names (latency and
    cost, which the run level computes directly) fall back to the
    ``n``-weighted mean of the values.

    Args:
        results: Rows sharing one ``name``.

    Returns:
        The aggregated row; ``n`` is the summed ``n``.

    Raises:
        ValueError: If ``results`` is empty or mixes names.
    """
    names = {result.name for result in results}
    if len(names) != 1:
        msg = f"combine_results needs rows of exactly one metric name, got {sorted(names)}"
        raise ValueError(msg)
    name = names.pop()
    n = sum(result.n for result in results)
    extras = [result.extra for result in results]
    if name == "detection_prf1":
        return detection_result(name, _sum_families(extras))
    if name == "route_prf1":
        stats = prf1_counts(*(int(_sum(extras, key)) for key in ("tp", "fp", "fn")))
        return MetricResult(name=name, value=stats["f1"], n=n, extra=stats)
    if name in {"label_cer", "label_wer"}:
        errors, ref_len = _sum(extras, "errors"), _sum(extras, "ref_len")
        extra = {"errors": errors, "ref_len": ref_len}
        return MetricResult(name=name, value=ratio(errors, ref_len), n=n, extra=extra)
    if name == "state_accuracy":
        return _combine_state(extras, n)
    if name == "topology_agreement":
        return _combine_topology(extras, n)
    if name == "calibration_brier":
        sum_sq = _sum(extras, "sum_sq_error")
        return MetricResult(name=name, value=ratio(sum_sq, n), n=n, extra={"sum_sq_error": sum_sq})
    if name == "calibration_ece":
        return ece_result(_sum_bins(extras), n)
    value = ratio(sum(result.value * result.n for result in results), n)
    return MetricResult(name=name, value=value, n=n)


def _sum(extras: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(extra.get(key, 0) for extra in extras))


def _sum_families(extras: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = {}
    for extra in extras:
        for family, stats in cast("Mapping[str, Mapping[str, float]]", extra["families"]).items():
            cell = counts.setdefault(family, {"tp": 0, "fp": 0, "fn": 0})
            for key in cell:
                cell[key] += int(stats[key])
    return {family: prf1_counts(c["tp"], c["fp"], c["fn"]) for family, c in counts.items()}


def _combine_state(extras: Sequence[Mapping[str, Any]], n: int) -> MetricResult:
    fields: dict[str, dict[str, int]] = {}
    for extra in extras:
        for field, stats in cast("Mapping[str, Mapping[str, int]]", extra["fields"]).items():
            cell = fields.setdefault(field, {"correct": 0, "total": 0})
            cell["correct"] += int(stats["correct"])
            cell["total"] += int(stats["total"])
    correct, total = int(_sum(extras, "correct")), int(_sum(extras, "total"))
    return MetricResult(
        name="state_accuracy",
        value=ratio(correct, total),
        n=n,
        extra={"correct": correct, "total": total, "fields": fields},
    )


def _combine_topology(extras: Sequence[Mapping[str, Any]], n: int) -> MetricResult:
    keys = ("intersection", "union", "gt_edges", "pred_edges", "degree_agree", "degree_total")
    sums = {key: int(_sum(extras, key)) for key in keys}
    value = sums["intersection"] / sums["union"] if sums["union"] else 1.0
    extra: dict[str, Any] = {
        **sums,
        "degree_agreement": ratio(sums["degree_agree"], sums["degree_total"]),
    }
    return MetricResult(name="topology_agreement", value=value, n=n, extra=extra)


def _sum_bins(extras: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    total: list[dict[str, float]] = []
    for extra in extras:
        bins = cast("Sequence[Mapping[str, float]]", extra["bins"])
        if not total:
            total = [{"count": 0, "sum_conf": 0.0, "sum_correct": 0} for _ in bins]
        if len(bins) != len(total):
            msg = "cannot combine calibration rows with different bin counts"
            raise ValueError(msg)
        for cell, add in zip(total, bins, strict=True):
            for key in cell:
                cell[key] += add[key]
    return total


def aggregate_run(
    run_dir: Path,
    *,
    ground_truth: Mapping[str, SceneAnnotation] | None = None,
    metrics: Sequence[str] | None = None,
) -> Path:
    """Score a run's predictions and write ``metrics.parquet``.

    Reads ``run_dir / RUN_LAYOUT["predictions"]``, scores the latest attempt
    of every (model, mode, scene) whose scene id is in ``ground_truth`` with
    :func:`score_prediction` (a record without a parsed annotation scores as
    an empty prediction), micro-aggregates each per-scene metric per (model,
    mode) with :func:`combine_results`, adds the latency and cost rows of all
    records (every attempt) per (model, mode), and writes the table described
    in the module docstring to ``run_dir / RUN_LAYOUT["metrics"]``. Records of
    scenes without ground truth are only counted for latency and cost.

    Args:
        run_dir: The run directory.
        ground_truth: Ground-truth documents keyed by scene id; None scores
            nothing but latency and cost.
        metrics: Names from :data:`METRIC_NAMES` to compute; all when None.

    Returns:
        The written metrics file.

    Raises:
        ValueError: If ``metrics`` contains an unknown name.
    """
    wanted = tuple(METRIC_NAMES if metrics is None else metrics)
    unknown = sorted(set(wanted) - set(METRIC_NAMES))
    if unknown:
        msg = f"unknown metric names {unknown}; expected a subset of {METRIC_NAMES}"
        raise ValueError(msg)
    # Imported here: the runner orchestrates evaluation and may import this module.
    from rail_vision_bench.runner import RUN_LAYOUT

    records = read_predictions(run_dir / RUN_LAYOUT["predictions"])
    truth = ground_truth or {}
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[PredictionRecord]] = {}
    for record in records:
        groups.setdefault((record.model, str(record.mode)), []).append(record)
    for (model, mode), group in groups.items():
        per_name: dict[str, list[MetricResult]] = {}
        for record in latest_attempts(group):
            gt = truth.get(record.scene_id)
            if gt is None:
                continue
            for result in score_prediction(gt, record.annotation, wanted):
                per_name.setdefault(result.name, []).append(result)
                rows.append(_row(model, mode, "scene", record.scene_id, result))
        run_rows = [combine_results(results) for results in per_name.values()]
        run_rows += [r for r in latency_summary(group) if r.name in wanted]
        order = {name: i for i, name in enumerate(METRIC_NAMES)}
        rows.extend(
            _row(model, mode, "run", None, result)
            for result in sorted(run_rows, key=lambda r: order[r.name])
        )
    path = run_dir / RUN_LAYOUT["metrics"]
    _write_parquet(rows, path)
    return path


def _row(
    model: str, mode: str, scope: str, scene_id: str | None, result: MetricResult
) -> dict[str, Any]:
    return {
        "model": model,
        "mode": mode,
        "scope": scope,
        "scene_id": scene_id,
        "name": result.name,
        "value": float(result.value),
        "n": int(result.n),
        "extra": json.dumps(result.extra, sort_keys=True),
    }


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    schema = pa.schema(
        [
            ("model", pa.string()),
            ("mode", pa.string()),
            ("scope", pa.string()),
            ("scene_id", pa.string()),
            ("name", pa.string()),
            ("value", pa.float64()),
            ("n", pa.int64()),
            ("extra", pa.string()),
        ]
    )
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    pq.write_table(table, path)


def load_metrics(path: Path, *, parse_extra: bool = True) -> Any:
    """Read a ``metrics.parquet`` written by :func:`aggregate_run`.

    Args:
        path: The parquet file.
        parse_extra: Decode the ``extra`` JSON strings into dicts.

    Returns:
        A ``pandas.DataFrame`` with the columns of :data:`COLUMNS` (typed as
        ``Any`` because pandas ships no type information here).
    """
    import pandas as pd  # type: ignore[import-untyped]

    frame = pd.read_parquet(path)
    if parse_extra:
        frame["extra"] = frame["extra"].map(json.loads)
    return frame
