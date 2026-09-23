"""Metric definitions and their per-scene computations.

Row scheme: every metric function returns :class:`MetricResult` rows whose
``name`` is one of :data:`METRIC_NAMES` -- exactly one row per name. ``value``
is the headline number, ``n`` the number of items it was computed over and
``extra`` holds the raw counts from which the value can be recomputed after
summing over scenes (micro-averaging, see
:func:`rail_vision_bench.eval.aggregate.combine_results`) plus any breakdown:

=================== =================================== ============================================
name                value                               ``extra``
=================== =================================== ============================================
detection_prf1      micro F1 over nodes, edges,         ``tp``/``fp``/``fn``/``precision``/
                    signals, derailers                  ``recall``/``f1`` and ``families`` ->
                                                        the same keys per family
label_cer           character error rate                ``errors``, ``ref_len``
label_wer           word error rate                     ``errors``, ``ref_len``
state_accuracy      correct / scored state items        ``correct``, ``total`` and ``fields`` ->
                                                        ``{correct, total}`` per state field
route_prf1          route F1                            ``tp``/``fp``/``fn``/``precision``/
                                                        ``recall``/``f1``
topology_agreement  port-exact edge Jaccard index       ``intersection``, ``union``, ``gt_edges``,
                                                        ``pred_edges``, ``degree_agree``,
                                                        ``degree_total``, ``degree_agreement``
calibration_brier   Brier score                         ``sum_sq_error``
calibration_ece     expected calibration error          ``bins``: ``count``/``sum_conf``/
                                                        ``sum_correct`` per equal-width bin
latency_p50_ms      median ``latency_ms``               ``mean``, ``min``, ``max``
latency_p95_ms      95th percentile of ``latency_ms``   ``mean``, ``min``, ``max``
cost_usd            total ``cost_usd``                  ``n_with_cost``
=================== =================================== ============================================

Empty inputs never divide by zero and never produce NaN:

* precision, recall and F1 with ``tp = fp = fn = 0`` are 1.0 (an empty
  prediction agrees perfectly with an empty ground truth); otherwise a zero
  denominator gives 0.0 (e.g. precision of an empty prediction);
* the edge Jaccard index of two empty edge sets is 1.0 for the same reason;
* every other rate (error rates, accuracy, calibration, latency, cost) is
  reported as 0.0 with ``n = 0`` -- ``n = 0`` marks the value as undefined,
  and aggregation weights by the raw counts, so such rows contribute nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import jiwer
import networkx as nx

from rail_vision_bench.eval.matching import FAMILIES, Family, Match, match_elements
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.graph.build import to_networkx
from rail_vision_bench.schema.models import (
    NodeKind,
    Occupancy,
    Port,
    SceneAnnotation,
    SwitchPosition,
)

METRIC_NAMES: Final[tuple[str, ...]] = (
    "detection_prf1",
    "label_cer",
    "label_wer",
    "state_accuracy",
    "route_prf1",
    "topology_agreement",
    "calibration_brier",
    "calibration_ece",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_usd",
)
"""Every metric name a task may request and a leaderboard may show."""

DETECTION_FAMILIES: Final[tuple[Family, ...]] = ("nodes", "edges", "signals", "derailers")
"""Families counted by ``detection_prf1``; routes have their own ``route_prf1``."""

RUN_LEVEL_METRICS: Final[frozenset[str]] = frozenset(
    {"latency_p50_ms", "latency_p95_ms", "cost_usd"}
)
"""Metrics computed from the prediction records of a run rather than per scene."""


def prf1_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from raw counts under the module's empty convention.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        ``tp``, ``fp``, ``fn``, ``precision``, ``recall`` and ``f1``.
    """
    if tp == fp == fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def ratio(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 for a zero denominator (the ``n = 0`` convention).

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        ``numerator / denominator``, or 0.0 when the denominator is 0.
    """
    return numerator / denominator if denominator else 0.0


def family_of(match: Match, gt: SceneAnnotation) -> Family | None:
    """Return the family of a match, looking the ground-truth id up when unset.

    Args:
        match: An accepted pair.
        gt: The ground-truth document.

    Returns:
        The family, or None when the ground-truth id is unknown.
    """
    if match.family is not None:
        return match.family
    lookups: dict[Family, Mapping[str, object]] = {
        "nodes": gt.node_map(),
        "edges": gt.edge_map(),
        "signals": gt.signal_map(),
        "derailers": gt.derailer_map(),
        "routes": gt.route_map(),
    }
    return next((family for family in FAMILIES if match.gt_id in lookups[family]), None)


def pairs_by_family(gt: SceneAnnotation, matches: Sequence[Match]) -> dict[Family, dict[str, str]]:
    """Group matches into ground-truth id -> predicted id maps per family.

    Duplicate ground-truth or predicted ids keep their first pair, so a
    hand-built match list can never count one element twice.

    Args:
        gt: The ground-truth document.
        matches: Accepted pairs of any families.

    Returns:
        Ground-truth id to predicted id, per family.
    """
    grouped: dict[Family, dict[str, str]] = {family: {} for family in FAMILIES}
    used: dict[Family, set[str]] = {family: set() for family in FAMILIES}
    for match in matches:
        family = family_of(match, gt)
        if family is None or match.gt_id in grouped[family] or match.pred_id in used[family]:
            continue
        grouped[family][match.gt_id] = match.pred_id
        used[family].add(match.pred_id)
    return grouped


def _family_size(doc: SceneAnnotation, family: Family) -> int:
    sizes: dict[Family, int] = {
        "nodes": len(doc.topology.nodes),
        "edges": len(doc.topology.edges),
        "signals": len(doc.signals),
        "derailers": len(doc.derailers),
        "routes": len(doc.routes),
    }
    return sizes[family]


def detection_prf1(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Precision, recall and F1 of element detection per family.

    Matched pairs are true positives, unmatched predictions false positives
    and unmatched ground-truth elements false negatives, counted separately
    for nodes, edges, signals and derailers (``extra["families"]``) and
    micro-averaged into the headline ``detection_prf1`` value (F1).

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs from :func:`rail_vision_bench.eval.matching.match_elements`.

    Returns:
        One ``detection_prf1`` row with ``n = tp + fp + fn``.
    """
    grouped = pairs_by_family(gt, matches)
    families: dict[str, dict[str, float]] = {}
    for family in DETECTION_FAMILIES:
        tp = len(grouped[family])
        families[family] = prf1_counts(
            tp, _family_size(pred, family) - tp, _family_size(gt, family) - tp
        )
    return [detection_result("detection_prf1", families)]


def detection_result(name: str, families: Mapping[str, Mapping[str, float]]) -> MetricResult:
    """Build a micro-averaged PRF1 row from per-family counts.

    Args:
        name: The metric name.
        families: ``tp``/``fp``/``fn`` (and derived rates) per family.

    Returns:
        The row, with the micro counts and rates at the top level of ``extra``.
    """
    tp, fp, fn = (int(sum(stats[key] for stats in families.values())) for key in ("tp", "fp", "fn"))
    micro = prf1_counts(tp, fp, fn)
    return MetricResult(
        name=name,
        value=micro["f1"],
        n=tp + fp + fn,
        extra={**micro, "families": {family: dict(stats) for family, stats in families.items()}},
    )


def label_cer_wer(pairs: Sequence[tuple[str, str]]) -> list[MetricResult]:
    """Character and word error rate of transcribed labels.

    ``jiwer`` aligns every (ground truth, prediction) pair; the rates are
    total edit operations over total reference length (characters for
    ``label_cer``, whitespace-separated words for ``label_wer``). Pairs whose
    ground-truth label is blank are skipped (there is nothing to transcribe);
    a missing prediction should be passed as ``""`` and counts as deletions.

    Args:
        pairs: ``(ground_truth_label, predicted_label)`` per matched element.

    Returns:
        ``label_cer`` and ``label_wer`` rows with ``n`` = scored pairs.
    """
    kept = [(ref.strip(), hyp.strip()) for ref, hyp in pairs if ref.strip()]
    if not kept:
        return [
            MetricResult(name=name, value=0.0, n=0, extra={"errors": 0, "ref_len": 0})
            for name in ("label_cer", "label_wer")
        ]
    refs = [ref for ref, _ in kept]
    hyps = [hyp for _, hyp in kept]
    chars = jiwer.process_characters(refs, hyps)
    words = jiwer.process_words(refs, hyps)
    rows = []
    for name, out in (("label_cer", chars), ("label_wer", words)):
        errors = out.substitutions + out.deletions + out.insertions
        ref_len = out.substitutions + out.deletions + out.hits
        rows.append(
            MetricResult(
                name=name,
                value=ratio(errors, ref_len),
                n=len(kept),
                extra={"errors": errors, "ref_len": ref_len},
            )
        )
    return rows


@dataclass(frozen=True)
class StateItem:
    """One scored state field of a matched element."""

    field: str
    correct: bool
    confidence: float | None


def state_items(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[StateItem]:
    """Compare the state of every matched element field by field.

    Scored fields: ``switch_position`` (plain switches), ``slip_paths`` (the
    set of unordered active port pairs of a dkw/ekw), ``occupancy`` and
    ``route_set`` (tracks), ``aspect`` (signals), ``derailer_position`` and
    ``route_status``. A field is scored only when the ground truth knows it
    (not ``unknown``/None); a missing predicted state entry or an ``unknown``
    prediction is incorrect.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        The scored items, each with the predicted state's confidence.
    """
    grouped = pairs_by_family(gt, matches)
    items: list[StateItem] = []
    gt_nodes = gt.node_map()
    for gt_id, pred_id in grouped["nodes"].items():
        g_sw = gt.state.switches.get(gt_id)
        node = gt_nodes.get(gt_id)
        if g_sw is None or node is None:
            continue
        p_sw = pred.state.switches.get(pred_id)
        conf = p_sw.confidence if p_sw else None
        if node.kind == NodeKind.SWITCH:
            if g_sw.position not in (None, SwitchPosition.UNKNOWN):
                ok = p_sw is not None and p_sw.position == g_sw.position
                items.append(StateItem("switch_position", ok, conf))
        elif g_sw.active_paths is not None:
            predicted = p_sw.active_paths if p_sw is not None else None
            ok = predicted is not None and _paths(predicted) == _paths(g_sw.active_paths)
            items.append(StateItem("slip_paths", ok, conf))
    for gt_id, pred_id in grouped["edges"].items():
        g_tr = gt.state.tracks.get(gt_id)
        if g_tr is None:
            continue
        p_tr = pred.state.tracks.get(pred_id)
        conf = p_tr.confidence if p_tr else None
        if g_tr.occupancy != Occupancy.UNKNOWN:
            items.append(
                StateItem("occupancy", p_tr is not None and p_tr.occupancy == g_tr.occupancy, conf)
            )
        if g_tr.route_set is not None:
            items.append(
                StateItem("route_set", p_tr is not None and p_tr.route_set == g_tr.route_set, conf)
            )
    items.extend(_simple_state(gt, pred, grouped))
    return items


def _paths(paths: Sequence[tuple[Port, Port]]) -> frozenset[frozenset[Port]]:
    return frozenset(frozenset(path) for path in paths)


def _simple_state(
    gt: SceneAnnotation, pred: SceneAnnotation, grouped: Mapping[Family, Mapping[str, str]]
) -> list[StateItem]:
    """Score the single-valued states of signals, derailers and routes."""
    items: list[StateItem] = []
    tables: tuple[tuple[Family, str, Mapping[str, Any], Mapping[str, Any], str], ...] = (
        ("signals", "aspect", gt.state.signals, pred.state.signals, "aspect"),
        ("derailers", "derailer_position", gt.state.derailers, pred.state.derailers, "position"),
        ("routes", "route_status", gt.state.routes, pred.state.routes, "status"),
    )
    for family, field, gt_table, pred_table, attr in tables:
        for gt_id, pred_id in grouped[family].items():
            g_state = gt_table.get(gt_id)
            if g_state is None or getattr(g_state, attr) == "unknown":
                continue
            p_state = pred_table.get(pred_id)
            ok = p_state is not None and getattr(p_state, attr) == getattr(g_state, attr)
            conf = cast("float | None", p_state.confidence) if p_state is not None else None
            items.append(StateItem(field, ok, conf))
    return items


def state_accuracy(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Accuracy of the observed state on matched elements.

    Compares switch positions and active paths, signal aspects, track
    occupancy and route illumination, derailer positions and route status of
    every matched pair (see :func:`state_items`); ``unknown`` never counts as
    correct. Unmatched elements are not scored here -- detection penalises
    them.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        One ``state_accuracy`` row with ``n`` = scored fields.
    """
    return [state_result(state_items(gt, pred, matches))]


def state_result(items: Sequence[StateItem]) -> MetricResult:
    """Build the ``state_accuracy`` row from scored items.

    Args:
        items: The scored state fields.

    Returns:
        The row with overall and per-field counts.
    """
    fields: dict[str, dict[str, int]] = {}
    for item in items:
        stats = fields.setdefault(item.field, {"correct": 0, "total": 0})
        stats["correct"] += int(item.correct)
        stats["total"] += 1
    correct = sum(int(item.correct) for item in items)
    return MetricResult(
        name="state_accuracy",
        value=ratio(correct, len(items)),
        n=len(items),
        extra={"correct": correct, "total": len(items), "fields": fields},
    )


def route_prf1(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Precision, recall and F1 of routes as edge sequences.

    Every predicted route is translated into ground-truth ids: its start and
    end through the signal (else node) matches, every path edge through the
    edge matches. A predicted route is a true positive when its translation
    equals (start, end and the ordered edge path) a not yet claimed
    ground-truth route; the remaining predicted routes are false positives and
    the unclaimed ground-truth routes false negatives. Route labels and
    ``switch_positions`` are not compared.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        One ``route_prf1`` row with ``n = tp + fp + fn``.
    """
    grouped = pairs_by_family(gt, matches)
    to_gt = {
        family: {pred_id: gt_id for gt_id, pred_id in pairs.items()}
        for family, pairs in grouped.items()
    }
    unclaimed = Counter((route.start, route.end, tuple(route.path)) for route in gt.routes)
    tp = 0
    for route in pred.routes:
        start = to_gt["signals"].get(route.start) or to_gt["nodes"].get(route.start)
        end = to_gt["signals"].get(route.end) or to_gt["nodes"].get(route.end)
        path = tuple(to_gt["edges"].get(edge) for edge in route.path)
        if start is None or end is None or any(edge is None for edge in path):
            continue
        key = (start, end, tuple(edge for edge in path if edge is not None))
        if unclaimed[key] > 0:
            unclaimed[key] -= 1
            tp += 1
    stats = prf1_counts(tp, len(pred.routes) - tp, len(gt.routes) - tp)
    return [
        MetricResult(
            name="route_prf1",
            value=stats["f1"],
            n=int(stats["tp"] + stats["fp"] + stats["fn"]),
            extra=stats,
        )
    ]


def topology_agreement(gt: SceneAnnotation, pred: SceneAnnotation) -> list[MetricResult]:
    """Graph-level agreement between the two topologies.

    The node matching is computed here with the default weights of
    :func:`match_elements`. Both topologies become multigraphs
    (:func:`rail_vision_bench.graph.build.to_networkx`); every edge is keyed
    by its unordered pair of ``(node, port)`` endpoints, predicted nodes being
    renamed to their matched ground-truth ids (unmatched ones stay distinct).
    The headline value is the port-exact Jaccard index of the two edge
    multisets; ``extra`` also reports the degree agreement: the fraction of
    ground-truth nodes whose matched predicted node has the same degree
    (unmatched nodes disagree).

    Args:
        gt: The ground-truth document.
        pred: The predicted document.

    Returns:
        One ``topology_agreement`` row with ``n`` = size of the edge union.
    """
    node_map = {m.pred_id: m.gt_id for m in match_elements(gt, pred, family="nodes")}
    gt_graph = to_networkx(gt.topology)
    pred_graph = to_networkx(pred.topology)
    gt_edges = _edge_keys(gt, gt_graph, None)
    pred_edges = _edge_keys(pred, pred_graph, node_map)
    intersection = sum((gt_edges & pred_edges).values())
    union = sum(gt_edges.values()) + sum(pred_edges.values()) - intersection
    jaccard = intersection / union if union else 1.0
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in node_map.items()}
    degree_agree = sum(
        1
        for node in gt_graph.nodes
        if node in gt_to_pred and pred_graph.degree(gt_to_pred[node]) == gt_graph.degree(node)
    )
    degree_total = gt_graph.number_of_nodes()
    return [
        MetricResult(
            name="topology_agreement",
            value=jaccard,
            n=union,
            extra={
                "intersection": intersection,
                "union": union,
                "gt_edges": sum(gt_edges.values()),
                "pred_edges": sum(pred_edges.values()),
                "degree_agree": degree_agree,
                "degree_total": degree_total,
                "degree_agreement": ratio(degree_agree, degree_total),
            },
        )
    ]


_EdgeKey = frozenset[tuple[str, str]]


def _edge_keys(
    doc: SceneAnnotation, graph: nx.MultiGraph[str], node_map: Mapping[str, str] | None
) -> Counter[_EdgeKey]:
    """Key every graph edge by its port-exact endpoints.

    With a ``node_map`` (predicted -> ground-truth id) predicted nodes are
    renamed and unmatched ones get a name no ground-truth node can have;
    without one the ids are kept. The ports come from the document's edges
    because an undirected multigraph does not preserve which end is ``a``.
    """
    keys: Counter[_EdgeKey] = Counter()
    for edge in doc.topology.edges:
        if edge.a.node in graph and edge.b.node in graph:
            ends = ((edge.a.node, str(edge.a.port)), (edge.b.node, str(edge.b.port)))
            keys[frozenset((_rename(node, node_map), port) for node, port in ends)] += 1
    return keys


def _rename(node: str, node_map: Mapping[str, str] | None) -> str:
    if node_map is None:
        return node
    return node_map.get(node, f"\0pred:{node}")


def calibration(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10
) -> list[MetricResult]:
    """Confidence calibration of the model's self-reported confidences.

    Brier score (``sklearn.metrics.brier_score_loss``) and expected
    calibration error over ``n_bins`` equal-width bins ``[k/n, (k+1)/n)``
    (the last bin includes 1.0): ``sum_b |B|/N * |accuracy(B) - mean confidence(B)|``.

    Args:
        confidences: Reported confidence per element in ``[0, 1]``.
        correct: Whether the corresponding element was right.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ``calibration_brier`` and ``calibration_ece`` rows with ``n`` = items.

    Raises:
        ValueError: If the sequences differ in length, ``n_bins < 1`` or a
            confidence lies outside ``[0, 1]``.
    """
    if len(confidences) != len(correct):
        msg = f"{len(confidences)} confidences but {len(correct)} outcomes"
        raise ValueError(msg)
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)
    if any(not 0.0 <= c <= 1.0 for c in confidences):
        msg = "confidences must lie in [0, 1]"
        raise ValueError(msg)
    n = len(confidences)
    bins = [{"count": 0, "sum_conf": 0.0, "sum_correct": 0} for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct, strict=True):
        cell = bins[min(int(conf * n_bins), n_bins - 1)]
        cell["count"] += 1
        cell["sum_conf"] += conf
        cell["sum_correct"] += int(ok)
    brier = 0.0
    if n:
        from sklearn.metrics import brier_score_loss  # type: ignore[import-untyped]

        brier = float(brier_score_loss([int(ok) for ok in correct], list(confidences), pos_label=1))
    return [
        MetricResult(name="calibration_brier", value=brier, n=n, extra={"sum_sq_error": brier * n}),
        ece_result(bins, n),
    ]


def ece_result(bins: Sequence[Mapping[str, float]], n: int) -> MetricResult:
    """Build the ``calibration_ece`` row from per-bin sums.

    Args:
        bins: ``count``, ``sum_conf`` and ``sum_correct`` per bin.
        n: Total number of items.

    Returns:
        The row, with the bins in ``extra``.
    """
    ece = sum(
        abs(cell["sum_correct"] - cell["sum_conf"]) / n for cell in bins if cell["count"] and n
    )
    return MetricResult(
        name="calibration_ece", value=float(ece), n=n, extra={"bins": [dict(b) for b in bins]}
    )


def latency_summary(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    """Latency percentiles and total cost of a run.

    ``latency_p50_ms`` and ``latency_p95_ms`` are linear-interpolation
    percentiles (``numpy.percentile``) of ``latency_ms`` over all records;
    ``cost_usd`` sums the known costs (records without a cost count as 0 and
    ``extra["n_with_cost"]`` says how many had one).

    Args:
        records: Every prediction record of the run.

    Returns:
        ``latency_p50_ms``, ``latency_p95_ms`` and ``cost_usd`` rows with
        ``n`` = number of records.
    """
    import numpy as np

    n = len(records)
    latencies = np.asarray([record.latency_ms for record in records], dtype=float)
    stats: dict[str, Any] = {"mean": 0.0, "min": 0.0, "max": 0.0}
    p50 = p95 = 0.0
    if n:
        p50, p95 = (float(v) for v in np.percentile(latencies, [50.0, 95.0]))
        stats = {
            "mean": float(latencies.mean()),
            "min": float(latencies.min()),
            "max": float(latencies.max()),
        }
    costs = [record.cost_usd for record in records if record.cost_usd is not None]
    return [
        MetricResult(name="latency_p50_ms", value=p50, n=n, extra=dict(stats)),
        MetricResult(name="latency_p95_ms", value=p95, n=n, extra=dict(stats)),
        MetricResult(
            name="cost_usd", value=float(sum(costs)), n=n, extra={"n_with_cost": len(costs)}
        ),
    ]
