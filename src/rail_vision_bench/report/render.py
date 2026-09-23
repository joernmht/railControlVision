"""Leaderboard rendering: rank models per metric and write a self-contained HTML or Markdown file.

The input is plain data so the caller decides where it comes from: metric rows
shaped like the columns of a run's ``metrics.parquet`` (``model``, ``mode``,
``name``, ``value``, ``n``, ``extra``; optional ``run_id`` and ``split``) and,
optionally, one mapping per run (the run's ``summary.json``). A DataFrame is
passed as ``df.to_dict("records")``.

Rows of one metric name are told apart by their *scope*: the string-valued
entries of ``extra`` (for example ``{"family": "signal", "stat": "f1"}``);
numeric and structured entries of ``extra`` are detail, not scope. Every
(metric, scope) gets its own ranked table; the overview shows one headline
scope per metric (see :func:`headline_scope`) and orders the entries by their
mean rank over those columns.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, Literal, TypeGuard

import orjson
from jinja2 import Environment, StrictUndefined

from rail_vision_bench.eval.metrics import METRIC_NAMES

TEMPLATE_NAME: Final = "leaderboard.html.j2"

DEFAULT_TITLE: Final = "rail-vision-bench leaderboard"

LOWER_IS_BETTER_TOKENS: Final[frozenset[str]] = frozenset(
    {"cer", "wer", "brier", "ece", "latency", "cost"}
)
"""Name tokens (split on ``_``) of metrics where a smaller value ranks higher.

Everything else (``prf1``, ``accuracy``, ``agreement``, ...) ranks higher-is-better.
"""

HEADLINE_SCOPE_VALUES: Final[frozenset[str]] = frozenset(
    {"all", "micro", "overall", "total", "mean", "f1"}
)
"""Scope values that mark an aggregate row eligible for the overview."""

Scope = tuple[tuple[str, str], ...]
"""The sorted string-valued ``extra`` entries of a metric row; ``()`` for an unscoped row."""

Direction = Literal["higher", "lower"]


def load_template() -> str:
    """Read the leaderboard template shipped inside the package.

    Returns:
        The Jinja2 template source.
    """
    resource = files("rail_vision_bench").joinpath("report", "templates", TEMPLATE_NAME)
    return resource.read_text(encoding="utf-8")


def metric_direction(name: str) -> Direction:
    """Tell whether a larger or a smaller value of a metric is better.

    Args:
        name: Metric name, e.g. ``detection_prf1`` or ``latency_p95_ms``.

    Returns:
        ``"lower"`` when a token of the name is in :data:`LOWER_IS_BETTER_TOKENS`,
        else ``"higher"``.
    """
    tokens = set(name.lower().split("_"))
    return "lower" if tokens & LOWER_IS_BETTER_TOKENS else "higher"


def row_scope(extra: object) -> Scope:
    """Extract the scope of a metric row from its ``extra``.

    Args:
        extra: The row's ``extra`` as a mapping or a JSON object string; anything
            else (``None``, NaN, an empty string) counts as no extra.

    Returns:
        The sorted ``(key, value)`` pairs whose value is a string.
    """
    return tuple(sorted((k, v) for k, v in _extra(extra).items() if isinstance(v, str)))


def scope_label(scope: Scope) -> str:
    """Render a scope for humans (``family=signal, stat=f1``; ``overall`` when empty).

    Args:
        scope: The scope.

    Returns:
        The label.
    """
    return ", ".join(f"{k}={v}" for k, v in scope) if scope else "overall"


def headline_scope(scopes: Iterable[Scope]) -> Scope | None:
    """Pick the scope of a metric shown in the overview.

    The unscoped row wins; otherwise the scope whose values are all in
    :data:`HEADLINE_SCOPE_VALUES` with the fewest keys (then the smallest
    label). A metric reported only per family has no headline.

    Args:
        scopes: The scopes present for one metric name.

    Returns:
        The headline scope, or ``None``.
    """
    candidates = sorted(
        (
            scope
            for scope in set(scopes)
            if all(value.lower() in HEADLINE_SCOPE_VALUES for _, value in scope)
        ),
        key=lambda scope: (len(scope), scope_label(scope)),
    )
    return candidates[0] if candidates else None


def rank(values: Sequence[float | None], direction: Direction) -> list[int | None]:
    """Competition ranking (1, 1, 3, ...) of values; missing or NaN values get no rank.

    Args:
        values: One value per entry.
        direction: Whether higher or lower is better.

    Returns:
        The rank of each value, in input order.
    """
    sign = -1.0 if direction == "higher" else 1.0
    present = sorted(sign * v for v in values if _is_number(v))
    ranks: list[int | None] = []
    for value in values:
        if _is_number(value):
            key = sign * value
            ranks.append(1 + sum(1 for other in present if other < key))
        else:
            ranks.append(None)
    return ranks


@dataclass(frozen=True, order=True)
class Entry:
    """One leaderboard contestant: a model in a mode, optionally per run and split."""

    model: str
    mode: str
    run_id: str = ""
    split: str = ""


@dataclass
class _Metric:
    name: str
    scope: Scope
    values: dict[Entry, tuple[float, int]] = field(default_factory=dict)


def render_leaderboard(
    metric_rows: Iterable[Mapping[str, Any]],
    out: Path,
    *,
    runs: Sequence[Mapping[str, Any]] = (),
    title: str = DEFAULT_TITLE,
    fmt: Literal["html", "md"] = "html",
    generated_at: datetime | None = None,
) -> Path:
    """Rank models per metric and write the leaderboard.

    Each metric row needs ``model``, ``mode``, ``name`` and ``value``; ``n``
    (default 0), ``extra`` (mapping, JSON string or ``None``), ``run_id`` and
    ``split`` are optional. Rows with the same entry, name and scope after the
    first are ignored. Ranking direction comes from :func:`metric_direction`.
    The HTML output is rendered from :data:`TEMPLATE_NAME` with autoescaping
    and carries its CSS inline (no external resources); ``md`` writes plain
    Markdown tables.

    Args:
        metric_rows: Metric rows of one or several runs.
        out: Destination file; parent directories are created.
        runs: Run metadata (e.g. each run's ``summary.json``), shown as a table;
            scalar values verbatim, others as compact JSON.
        title: Page title.
        fmt: Output format.
        generated_at: Timestamp printed in the header; now (UTC) when omitted.

    Returns:
        The written file.

    Raises:
        KeyError: When a metric row lacks a required column.
    """
    context = build_context(metric_rows, runs=runs, title=title, generated_at=generated_at)
    if fmt == "html":
        env = Environment(autoescape=True, undefined=StrictUndefined, keep_trailing_newline=True)
        text = env.from_string(load_template()).render(**context)
    else:
        text = _markdown(context)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def build_context(
    metric_rows: Iterable[Mapping[str, Any]],
    *,
    runs: Sequence[Mapping[str, Any]] = (),
    title: str = DEFAULT_TITLE,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Compute the template context of :func:`render_leaderboard` (exposed for testing).

    Args:
        metric_rows: Metric rows, see :func:`render_leaderboard`.
        runs: Run metadata mappings.
        title: Page title.
        generated_at: Header timestamp; now (UTC) when omitted.

    Returns:
        ``title``, ``generated_at``, ``show_run``, ``show_split``, ``overview``
        (``columns`` and ``rows``), ``rows`` (the overview rows, best mean rank
        first; each carries ``model``, ``mode``, ``run_id``, ``split``,
        ``position``, ``mean_rank`` and one ``cells`` item per overview
        column), ``metrics`` (one ranked table per metric and scope) and
        ``runs`` (``columns`` and ``rows``).

    Raises:
        KeyError: When a metric row lacks a required column.
    """
    metrics: dict[tuple[str, Scope], _Metric] = {}
    entries: set[Entry] = set()
    for row in metric_rows:
        entry = Entry(
            model=str(row["model"]),
            mode=str(row["mode"]),
            run_id=_opt_str(row.get("run_id")),
            split=_opt_str(row.get("split")),
        )
        entries.add(entry)
        name = str(row["name"])
        scope = row_scope(row.get("extra"))
        metric = metrics.setdefault((name, scope), _Metric(name, scope))
        n_raw = row.get("n")
        n = int(n_raw) if _is_number(n_raw) else 0
        value = row["value"]
        metric.values.setdefault(entry, (float(value) if _is_number(value) else math.nan, n))

    ordered = sorted(metrics.values(), key=lambda m: (_name_order(m.name), scope_label(m.scope)))
    tables = {(m.name, m.scope): _table(m) for m in ordered}

    # Overview: one headline scope per metric name.
    scopes_by_name: dict[str, list[Scope]] = {}
    for metric in ordered:
        scopes_by_name.setdefault(metric.name, []).append(metric.scope)
    columns: list[dict[str, Any]] = []
    for name, scopes in scopes_by_name.items():
        headline = headline_scope(scopes)
        if headline is not None:
            columns.append(
                {"name": name, "scope": scope_label(headline), "table": tables[(name, headline)]}
            )
    overview_rows: list[dict[str, Any]] = []
    for entry in entries:
        cells: list[dict[str, Any]] = []
        entry_ranks: list[int] = []
        for column in columns:
            cell = next((r for r in column["table"]["rows"] if r["entry"] == entry), None)
            if cell is None or cell["rank"] is None:
                cells.append({"value": "—", "rank": None, "best": False})
                continue
            entry_ranks.append(cell["rank"])
            cells.append({"value": cell["value"], "rank": cell["rank"], "best": cell["rank"] == 1})
        mean_rank = sum(entry_ranks) / len(entry_ranks) if entry_ranks else math.inf
        overview_rows.append(
            {**_entry_fields(entry), "entry": entry, "cells": cells, "mean_rank_value": mean_rank}
        )
    overview_rows.sort(key=lambda r: (r["mean_rank_value"], r["entry"]))
    for position, overview_row in enumerate(overview_rows, start=1):
        overview_row["position"] = position
        value = overview_row["mean_rank_value"]
        overview_row["mean_rank"] = f"{value:.2f}" if math.isfinite(value) else "—"

    return {
        "title": title,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(timespec="seconds"),
        "show_run": any(entry.run_id for entry in entries),
        "show_split": any(entry.split for entry in entries),
        "overview": {
            "columns": [
                {
                    "name": c["name"],
                    "scope": c["scope"],
                    "direction": metric_direction(c["name"]),
                }
                for c in columns
            ],
            "rows": overview_rows,
        },
        "rows": overview_rows,
        "metrics": list(tables.values()),
        "runs": _runs_table(runs),
    }


# ----------------------------------------------------------------------------- helpers


def _is_number(value: object) -> TypeGuard[float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and not (isinstance(value, float) and math.isnan(value))
    )


def _opt_str(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _extra(extra: object) -> Mapping[str, Any]:
    if isinstance(extra, str) and extra.strip():
        loaded = orjson.loads(extra)
        return loaded if isinstance(loaded, dict) else {}
    return extra if isinstance(extra, Mapping) else {}


def _name_order(name: str) -> tuple[int, str]:
    return (METRIC_NAMES.index(name), "") if name in METRIC_NAMES else (len(METRIC_NAMES), name)


def _entry_fields(entry: Entry) -> dict[str, str]:
    return {"model": entry.model, "mode": entry.mode, "run_id": entry.run_id, "split": entry.split}


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.4g}"


def _table(metric: _Metric) -> dict[str, Any]:
    direction = metric_direction(metric.name)
    entries = sorted(metric.values)
    values = [metric.values[e][0] for e in entries]
    ranks = rank(values, direction)
    finite = [abs(v) for v in values if not math.isnan(v)]
    scale = max(finite, default=0.0)
    rows = [
        {
            **_entry_fields(entry),
            "entry": entry,
            "rank": entry_rank,
            "value": _fmt(value),
            "n": metric.values[entry][1],
            "bar": round(100 * abs(value) / scale, 1) if scale and not math.isnan(value) else 0.0,
        }
        for entry, value, entry_rank in zip(entries, values, ranks, strict=True)
    ]
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["entry"]))
    return {
        "name": metric.name,
        "scope": scope_label(metric.scope),
        "direction": direction,
        "rows": rows,
    }


def _cell(value: object) -> str:
    if value is None or isinstance(value, str | int | float | bool):
        return "" if value is None else str(value)
    return orjson.dumps(value, default=str).decode("utf-8")


def _runs_table(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns: list[str] = []
    for run in runs:
        columns.extend(key for key in run if key not in columns)
    return {
        "columns": columns,
        "rows": [[_cell(run.get(column)) for column in columns] for run in runs],
    }


def _md_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    def esc(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(esc(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(esc(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _entry_cells(entry: Entry, context: Mapping[str, Any]) -> list[str]:
    cells = [entry.model, entry.mode]
    if context["show_run"]:
        cells.append(entry.run_id)
    if context["show_split"]:
        cells.append(entry.split)
    return cells


def _markdown(context: Mapping[str, Any]) -> str:
    entry_header = ["model", "mode"]
    if context["show_run"]:
        entry_header.append("run")
    if context["show_split"]:
        entry_header.append("split")
    arrow = {"higher": "↑", "lower": "↓"}
    parts = [f"# {context['title']}", "", f"Generated at {context['generated_at']}", ""]
    overview = context["overview"]
    parts += [
        "## Overview",
        "",
        _md_table(
            ["#", *entry_header, "mean rank"]
            + [f"{c['name']} ({c['scope']}) {arrow[c['direction']]}" for c in overview["columns"]],
            (
                [r["position"], *_entry_cells(r["entry"], context), r["mean_rank"]]
                + [c["value"] for c in r["cells"]]
                for r in overview["rows"]
            ),
        ),
        "",
    ]
    for table in context["metrics"]:
        parts += [
            f"## {table['name']} ({table['scope']}) {arrow[table['direction']]}",
            "",
            _md_table(
                ["rank", *entry_header, "value", "n"],
                (
                    [r["rank"] or "—", *_entry_cells(r["entry"], context), r["value"], r["n"]]
                    for r in table["rows"]
                ),
            ),
            "",
        ]
    runs = context["runs"]
    if runs["rows"]:
        parts += ["## Runs", "", _md_table(runs["columns"], runs["rows"]), ""]
    return "\n".join(parts)
