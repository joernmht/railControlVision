"""Leaderboard ranking, the HTML/Markdown rendering and its escaping."""

from __future__ import annotations

import html
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rail_vision_bench.eval.metrics import METRIC_NAMES
from rail_vision_bench.report.render import (
    build_context,
    headline_scope,
    metric_direction,
    rank,
    render_leaderboard,
    row_scope,
    scope_label,
)

WHEN = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def _row(model: str, name: str, value: float, *, mode: str = "single_shot", **extra: Any):
    return {"model": model, "mode": mode, "name": name, "value": value, "n": 10, "extra": extra}


ROWS: list[dict[str, Any]] = [
    # detection_prf1: per-family rows plus a micro-averaged f1 as JSON string, as in parquet.
    *(
        {**_row(model, "detection_prf1", value), "extra": json.dumps({"family": fam, "stat": st})}
        for model, fam, st, value in [
            ("alpha", "micro", "f1", 0.80),
            ("beta", "micro", "f1", 0.90),
            ("alpha", "signals", "f1", 0.70),
            ("beta", "signals", "f1", 0.60),
            ("alpha", "micro", "precision", 0.85),
            ("beta", "micro", "precision", 0.95),
        ]
    ),
    _row("alpha", "label_cer", 0.10),
    _row("beta", "label_cer", 0.20, detail=[1, 2]),
    _row("alpha", "latency_p95_ms", 900.0),
    _row("beta", "latency_p95_ms", 1200.0),
    _row("alpha", "state_accuracy", float("nan")),
    _row("beta", "state_accuracy", 0.5),
]


@pytest.mark.parametrize(
    ("name", "direction"),
    [
        ("detection_prf1", "higher"),
        ("route_prf1", "higher"),
        ("state_accuracy", "higher"),
        ("topology_agreement", "higher"),
        ("label_cer", "lower"),
        ("label_wer", "lower"),
        ("calibration_brier", "lower"),
        ("calibration_ece", "lower"),
        ("latency_p50_ms", "lower"),
        ("latency_p95_ms", "lower"),
        ("cost_usd", "lower"),
    ],
)
def test_metric_direction(name: str, direction: str):
    assert name in METRIC_NAMES
    assert metric_direction(name) == direction


def test_rank_competition_and_missing():
    assert rank([0.5, 0.9, 0.5, None, math.nan], "higher") == [2, 1, 2, None, None]
    assert rank([3.0, 1.0, 2.0, 1.0], "lower") == [4, 1, 3, 1]
    assert rank([], "higher") == []


def test_scopes():
    assert row_scope(None) == ()
    assert row_scope(math.nan) == ()
    assert row_scope("") == ()
    assert row_scope('{"stat": "f1", "family": "all", "tp": 3}') == (
        ("family", "all"),
        ("stat", "f1"),
    )
    assert row_scope({"bins": [0.1], "count": 2}) == ()
    assert scope_label(()) == "overall"
    assert scope_label((("family", "all"), ("stat", "f1"))) == "family=all, stat=f1"
    assert headline_scope([(("family", "x"),), ()]) == ()
    assert headline_scope([(("stat", "f1"),), (("family", "micro"), ("stat", "f1"))]) == (
        ("stat", "f1"),
    )
    assert headline_scope([(("family", "signals"), ("stat", "f1"))]) is None


def test_build_context_ranks_per_metric():
    context = build_context(ROWS, generated_at=WHEN)
    assert context["generated_at"] == "2026-09-23T12:00:00+00:00"
    tables = {(t["name"], t["scope"]): t for t in context["metrics"]}
    order = [(r["model"], r["rank"], r["value"]) for r in tables[("label_cer", "overall")]["rows"]]
    assert order == [("alpha", 1, "0.1"), ("beta", 2, "0.2")]
    micro_f1 = tables[("detection_prf1", "family=micro, stat=f1")]["rows"]
    assert [(r["model"], r["rank"]) for r in micro_f1] == [("beta", 1), ("alpha", 2)]
    assert micro_f1[0]["bar"] == 100.0
    signals = tables[("detection_prf1", "family=signals, stat=f1")]["rows"]
    assert [r["model"] for r in signals] == ["alpha", "beta"]
    state = tables[("state_accuracy", "overall")]["rows"]
    assert [(r["model"], r["rank"], r["value"]) for r in state] == [
        ("beta", 1, "0.5"),
        ("alpha", None, "—"),
    ]
    # Metric tables follow METRIC_NAMES order.
    names = list(dict.fromkeys(t["name"] for t in context["metrics"]))
    assert names == ["detection_prf1", "label_cer", "state_accuracy", "latency_p95_ms"]

    columns = context["overview"]["columns"]
    assert [(c["name"], c["scope"]) for c in columns] == [
        ("detection_prf1", "family=micro, stat=f1"),
        ("label_cer", "overall"),
        ("state_accuracy", "overall"),
        ("latency_p95_ms", "overall"),
    ]
    rows = context["rows"]
    # alpha: ranks 2, 1, -, 1 -> 1.33; beta: 1, 2, 1, 2 -> 1.5
    assert [(r["position"], r["model"], r["mean_rank"]) for r in rows] == [
        (1, "alpha", "1.33"),
        (2, "beta", "1.50"),
    ]
    assert [c["value"] for c in rows[0]["cells"]] == ["0.8", "0.1", "—", "900"]
    assert context["show_run"] is False
    assert context["show_split"] is False


def test_duplicate_rows_keep_the_first():
    rows = [_row("a", "label_cer", 0.1), _row("a", "label_cer", 0.9)]
    (table,) = build_context(rows)["metrics"]
    assert [r["value"] for r in table["rows"]] == ["0.1"]


def test_missing_required_column_is_a_key_error():
    with pytest.raises(KeyError, match="value"):
        build_context([{"model": "a", "mode": "m", "name": "x"}])


def test_render_html_is_self_contained_and_escaped(tmp_path: Path):
    evil = "<script>alert(1)</script>"
    rows = [*ROWS, {**_row(evil, "label_cer", 0.3), "run_id": "r1", "split": "synthetic_clean"}]
    runs = [{"run_id": "r1", "models": ["alpha", "beta"], "note": evil, "n_scenes": 2}]
    out = render_leaderboard(
        rows,
        tmp_path / "nested" / "leaderboard.html",
        runs=runs,
        title="Bench & co",
        generated_at=WHEN,
    )
    assert out == tmp_path / "nested" / "leaderboard.html"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<script" not in text
    assert html.escape(evil) in text
    assert "<title>Bench &amp; co</title>" in text
    assert not re.search(r"""(src|href)=["']?(https?:)?//""", text)
    assert "<link" not in text
    assert "family=micro, stat=f1" in text
    assert "↓ lower is better" in text
    assert '<th class="text">run</th>' in text  # a row carries run_id
    assert "[&#34;alpha&#34;,&#34;beta&#34;]" in text  # non-scalar run value as JSON
    assert "2026-09-23T12:00:00+00:00" in text


def test_render_empty_html(tmp_path: Path):
    text = render_leaderboard([], tmp_path / "e.html", generated_at=WHEN).read_text()
    assert "No metric rows." in text


def test_render_markdown(tmp_path: Path):
    out = render_leaderboard(
        ROWS, tmp_path / "lb.md", fmt="md", runs=[{"run_id": "r|1"}], generated_at=WHEN
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# rail-vision-bench leaderboard\n")
    assert "## Overview" in text
    assert "| 1 | alpha | single_shot | 1.33 | 0.8 | 0.1 | — | 900 |" in text
    assert "## label_cer (overall) ↓" in text
    assert "| 1 | alpha | single_shot | 0.1 | 10 |" in text
    assert "## Runs" in text
    assert "r\\|1" in text
