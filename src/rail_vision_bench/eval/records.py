"""Row contracts of a run: one prediction per attempt and one metric per name."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.config import Mode
from rail_vision_bench.schema.issues import ValidationReport
from rail_vision_bench.schema.models import SceneAnnotation


class PredictionRecord(BaseModel):
    """One line of ``predictions.jsonl``: a model's attempt at one scene."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    run_id: str
    model: str
    mode: Mode
    attempt: int = 1
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    raw_text: str = Field(default="", description="The model output before parsing.")
    annotation: SceneAnnotation | None = Field(
        default=None, description="The parsed document, or None when parsing failed."
    )
    parse_error: str | None = None
    validation: ValidationReport | None = None


class MetricResult(BaseModel):
    """One metric value computed over ``n`` items."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    n: int
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Metric-specific detail (per-class counts, bins, ...)."
    )
