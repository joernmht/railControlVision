"""Row contracts of the evaluation stage and the metric name table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rail_vision_bench.config import Mode
from rail_vision_bench.eval.matching import Match
from rail_vision_bench.eval.metrics import METRIC_NAMES
from rail_vision_bench.eval.records import MetricResult, PredictionRecord
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.schema.models import SceneAnnotation


def test_prediction_record_round_trip(station_dkw_scene: SceneAnnotation):
    record = PredictionRecord(
        scene_id=station_dkw_scene.scene_id,
        run_id="run-1",
        model="m",
        mode=Mode.AGENTIC,
        attempt=2,
        latency_ms=12.5,
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.001,
        raw_text="{}",
        annotation=station_dkw_scene,
        validation=validate_document(station_dkw_scene, strict=True),
    )
    dumped = record.model_dump(mode="json")
    assert dumped["mode"] == "agentic"
    assert dumped["validation"]["ok"] is True
    assert PredictionRecord.model_validate(dumped) == record


def test_prediction_record_defaults_and_forbid():
    record = PredictionRecord(
        scene_id="s", run_id="r", model="m", mode=Mode.SINGLE_SHOT, latency_ms=1
    )
    assert record.attempt == 1
    assert record.raw_text == ""
    assert record.annotation is None
    assert record.parse_error is None
    assert record.validation is None
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**record.model_dump(), "extra": 1})


def test_metric_result_defaults():
    result = MetricResult(name="detection_prf1", value=0.5, n=3)
    assert result.extra == {}
    assert MetricResult.model_validate(result.model_dump(mode="json")) == result


def test_metric_names():
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))
    assert "detection_prf1" in METRIC_NAMES
    assert all(name == name.lower() for name in METRIC_NAMES)


def test_match_rejects_negative_cost():
    assert Match(gt_id="a", pred_id="b", cost=0.0).cost == 0.0
    with pytest.raises(ValidationError):
        Match(gt_id="a", pred_id="b", cost=-0.1)
