"""Metric definitions (names real, computations stubs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from rail_vision_bench.eval.matching import Match
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.schema.models import SceneAnnotation

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


def detection_prf1(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Precision, recall and F1 of element detection per family.

    Intended implementation: matched pairs are true positives, unmatched
    predictions false positives and unmatched ground-truth elements false
    negatives, reported per family and micro-averaged as ``detection_prf1``.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs from :func:`rail_vision_bench.eval.matching.match_elements`.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.detection_prf1 is not implemented in the skeleton: "
        "precision/recall/F1 of matched elements per family"
    )


def label_cer_wer(pairs: Sequence[tuple[str, str]]) -> list[MetricResult]:
    """Character and word error rate of transcribed labels.

    Intended implementation: ``jiwer.cer`` and ``jiwer.wer`` over the
    (ground truth, prediction) label pairs of matched elements, reported as
    ``label_cer`` and ``label_wer``.

    Args:
        pairs: ``(ground_truth_label, predicted_label)`` per matched element.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.label_cer_wer is not implemented in the skeleton: "
        "character and word error rates of matched labels with jiwer"
    )


def state_accuracy(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Accuracy of the observed state on matched elements.

    Intended implementation: compare switch positions and active paths,
    signal aspects, track occupancy and route illumination, derailer
    positions and route status of every matched pair; ``unknown`` never
    counts as correct.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.state_accuracy is not implemented in the skeleton: "
        "per-element-kind state accuracy on matched elements"
    )


def route_prf1(
    gt: SceneAnnotation, pred: SceneAnnotation, matches: Sequence[Match]
) -> list[MetricResult]:
    """Precision, recall and F1 of routes as edge sequences.

    Intended implementation: map predicted routes through the node/edge
    matches onto ground-truth ids and count a route as correct only when its
    start, end and edge path agree.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        matches: Accepted pairs of every family.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.route_prf1 is not implemented in the skeleton: "
        "precision/recall/F1 of routes translated through the element matches"
    )


def topology_agreement(gt: SceneAnnotation, pred: SceneAnnotation) -> list[MetricResult]:
    """Graph-level agreement between the two topologies.

    Intended implementation: build both multigraphs with
    :func:`rail_vision_bench.graph.build.to_networkx`, count port-exact edge
    agreement under the node matching and report the Jaccard index of the
    edge sets plus the degree-sequence agreement.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.topology_agreement is not implemented in the skeleton: "
        "port-exact edge agreement between the two networkx multigraphs"
    )


def calibration(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10
) -> list[MetricResult]:
    """Confidence calibration of the model's self-reported confidences.

    Intended implementation: Brier score (``sklearn.metrics.brier_score_loss``)
    and expected calibration error over ``n_bins`` equal-width bins, reported
    as ``calibration_brier`` and ``calibration_ece``.

    Args:
        confidences: Reported confidence per element in ``[0, 1]``.
        correct: Whether the corresponding element was right.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.calibration is not implemented in the skeleton: "
        "Brier score and expected calibration error of reported confidences"
    )


def latency_summary(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    """Latency percentiles and total cost of a run.

    Intended implementation: p50 and p95 of ``latency_ms`` and the sum of
    ``cost_usd`` over the records, reported as ``latency_p50_ms``,
    ``latency_p95_ms`` and ``cost_usd``.

    Args:
        records: Every prediction record of the run.

    Returns:
        The metric rows.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.metrics.latency_summary is not implemented in the skeleton: "
        "latency percentiles and total cost over the prediction records"
    )
