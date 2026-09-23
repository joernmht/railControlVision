"""Evaluation: prediction records, element matching, metrics and aggregation."""

from __future__ import annotations

from rail_vision_bench.eval.aggregate import aggregate_run, combine_results, load_metrics
from rail_vision_bench.eval.matching import Match, match_all, match_elements
from rail_vision_bench.eval.metrics import METRIC_NAMES
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.eval.scoring import score_prediction

__all__ = [
    "METRIC_NAMES",
    "Match",
    "MetricResult",
    "PredictionRecord",
    "aggregate_run",
    "combine_results",
    "load_metrics",
    "match_all",
    "match_elements",
    "score_prediction",
]
