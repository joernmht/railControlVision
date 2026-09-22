"""Wire models of the harness endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.config import Mode
from rail_vision_bench.schema.issues import ValidationIssue, ValidationReport
from rail_vision_bench.schema.models import SceneAnnotation


class HealthResponse(BaseModel):
    """``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    version: str
    schema_version: str


class ValidateResponse(BaseModel):
    """``POST /validate``: the validation report of the posted document."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class FrameResponse(BaseModel):
    """``POST /frame``: one inference over one uploaded frame."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    model: str
    mode: Mode
    annotation: SceneAnnotation | None = None
    validation: ValidationReport | None = None
    latency_ms: float
