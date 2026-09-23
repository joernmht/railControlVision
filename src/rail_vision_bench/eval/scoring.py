"""Per-scene scoring: match a prediction against its ground truth and run the metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rail_vision_bench.eval.matching import Family, Match, match_all
from rail_vision_bench.eval.metrics import (
    METRIC_NAMES,
    RUN_LEVEL_METRICS,
    StateItem,
    calibration,
    detection_prf1,
    label_cer_wer,
    pairs_by_family,
    route_prf1,
    state_items,
    state_result,
    topology_agreement,
)
from rail_vision_bench.eval.records import MetricResult
from rail_vision_bench.schema.models import (
    Provenance,
    ProvenanceKind,
    SceneAnnotation,
    State,
    Topology,
)


def empty_prediction(gt: SceneAnnotation) -> SceneAnnotation:
    """Return a prediction with no elements for the scene of ``gt``.

    Args:
        gt: The ground-truth document.

    Returns:
        A document with the same scene id and source and nothing else.
    """
    return SceneAnnotation(
        schema_version="v0",
        scene_id=gt.scene_id,
        source=gt.source,
        provenance=Provenance(kind=ProvenanceKind.PREDICTION),
        topology=Topology(),
        state=State(),
    )


def score_prediction(
    gt: SceneAnnotation,
    pred: SceneAnnotation | None,
    metrics: Sequence[str] = METRIC_NAMES,
) -> list[MetricResult]:
    """Match a prediction against its ground truth and compute the per-scene metrics.

    A missing prediction (``None``, e.g. a failed parse) is scored as an empty
    document, so every ground-truth element counts as missed. The run-level
    metrics (latency and cost) are skipped here; they come from the prediction
    records (:func:`rail_vision_bench.eval.metrics.latency_summary`).

    Label pairs for ``label_cer``/``label_wer`` are the labels of matched
    elements of every family whose ground-truth label is non-blank (the
    prediction's missing label counts as ``""``). Calibration pairs every
    confidence the prediction reports with an outcome: an element confidence
    is correct when the element is matched, a state confidence when the
    scored state field is correct (see
    :func:`rail_vision_bench.eval.metrics.state_items`).

    Args:
        gt: The ground-truth document.
        pred: The predicted document, or None when there is none.
        metrics: Names from :data:`METRIC_NAMES` to compute.

    Returns:
        One row per requested per-scene metric, in :data:`METRIC_NAMES` order.

    Raises:
        ValueError: If ``metrics`` contains an unknown name.
    """
    unknown = sorted(set(metrics) - set(METRIC_NAMES))
    if unknown:
        msg = f"unknown metric names {unknown}; expected a subset of {METRIC_NAMES}"
        raise ValueError(msg)
    wanted = set(metrics) - RUN_LEVEL_METRICS
    doc = pred if pred is not None else empty_prediction(gt)
    by_family = match_all(gt, doc)
    matches = [match for family_matches in by_family.values() for match in family_matches]
    items = state_items(gt, doc, matches)
    rows: list[MetricResult] = []
    if "detection_prf1" in wanted:
        rows += detection_prf1(gt, doc, matches)
    if wanted & {"label_cer", "label_wer"}:
        rows += label_cer_wer(label_pairs(gt, doc, matches))
    if "state_accuracy" in wanted:
        rows.append(state_result(items))
    if "route_prf1" in wanted:
        rows += route_prf1(gt, doc, matches)
    if "topology_agreement" in wanted:
        rows += topology_agreement(gt, doc)
    if wanted & {"calibration_brier", "calibration_ece"}:
        confidences, outcomes = confidence_outcomes(doc, by_family, items)
        rows += calibration(confidences, outcomes)
    order = {name: i for i, name in enumerate(METRIC_NAMES)}
    return sorted((row for row in rows if row.name in wanted), key=lambda row: order[row.name])


def label_pairs(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[tuple[str, str]]:
    """Collect ``(ground truth, prediction)`` label pairs of matched elements.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        One pair per matched element whose ground-truth label is non-blank.
    """
    labels = _labels(gt)
    pred_labels = _labels(pred)
    pairs: list[tuple[str, str]] = []
    for family, pairs_of_family in pairs_by_family(gt, matches).items():
        for gt_id, pred_id in pairs_of_family.items():
            gt_label = labels[family].get(gt_id)
            if gt_label and gt_label.strip():
                pairs.append((gt_label, pred_labels[family].get(pred_id) or ""))
    return pairs


def _labels(doc: SceneAnnotation) -> dict[Family, dict[str, str | None]]:
    return {
        "nodes": {node.id: node.label for node in doc.topology.nodes},
        "edges": {edge.id: edge.label for edge in doc.topology.edges},
        "signals": {signal.id: signal.label for signal in doc.signals},
        "derailers": {derailer.id: derailer.label for derailer in doc.derailers},
        "routes": {route.id: route.label for route in doc.routes},
    }


def confidence_outcomes(
    pred: SceneAnnotation,
    matches: Mapping[Family, Sequence[Match]],
    items: Sequence[StateItem],
) -> tuple[list[float], list[bool]]:
    """Pair every reported confidence of a prediction with its outcome.

    Args:
        pred: The predicted document.
        matches: Accepted pairs per family.
        items: Scored state fields (their confidence is the predicted state's).

    Returns:
        Confidences and whether each was right, in the same order.
    """
    matched = {family: {m.pred_id for m in pairs} for family, pairs in matches.items()}
    elements: tuple[tuple[Family, Sequence[tuple[str, float | None]]], ...] = (
        ("nodes", [(n.id, n.confidence) for n in pred.topology.nodes]),
        ("edges", [(e.id, e.confidence) for e in pred.topology.edges]),
        ("signals", [(s.id, s.confidence) for s in pred.signals]),
        ("derailers", [(d.id, d.confidence) for d in pred.derailers]),
        ("routes", [(r.id, r.confidence) for r in pred.routes]),
    )
    confidences: list[float] = []
    outcomes: list[bool] = []
    for family, entries in elements:
        for element_id, conf in entries:
            if conf is not None:
                confidences.append(conf)
                outcomes.append(element_id in matched.get(family, set()))
    for item in items:
        if item.confidence is not None:
            confidences.append(item.confidence)
            outcomes.append(item.correct)
    return confidences, outcomes
