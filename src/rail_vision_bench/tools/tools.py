"""The four agent tools.

``validate_tool`` is real because it only wraps the validator; the image tools
depend on rendering and OCR code that does not exist yet and are stubs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.graph.validator import validate_document

TOOL_NAMES: Final[tuple[str, ...]] = ("crop", "read", "validate", "render")
"""Tool names as exposed to agents and over MCP."""


class ToolResult(BaseModel):
    """Uniform tool outcome: success flag, structured payload and human-readable errors."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def crop_tool(image: bytes, bbox: tuple[float, float, float, float], *, zoom: float = 2.0) -> bytes:
    """Cut a region out of an image and upscale it for reading small labels.

    The implementation will decode ``image`` with Pillow, crop ``bbox``
    (``x0, y0, x1, y1`` in pixels), resize by ``zoom`` with Lanczos and return
    PNG bytes.

    Args:
        image: Encoded image bytes.
        bbox: The region in pixel coordinates.
        zoom: Upscaling factor applied to the crop.

    Returns:
        The encoded crop.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.tools.tools.crop_tool is not implemented in the skeleton: "
        "crop a bbox out of the image and upscale it"
    )


def read_tool(image: bytes, question: str) -> ToolResult:
    """Ask a focused question about an image crop (a label, a lamp, an indication).

    The implementation will send the crop and ``question`` to the current
    provider and return the transcription in ``payload["text"]`` together with
    the model's confidence.

    Args:
        image: Encoded crop bytes.
        question: What to read or decide, e.g. "Which digits follow 'W'?".

    Returns:
        The reading.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.tools.tools.read_tool is not implemented in the skeleton: "
        "answer a focused reading question about a crop with the provider"
    )


def validate_tool(document: Mapping[str, Any]) -> ToolResult:
    """Validate a candidate document against schema v0 and the semantic rules.

    Args:
        document: A raw ``SceneAnnotation`` JSON object.

    Returns:
        ``ok`` mirrors the report, ``payload`` is the report as JSON and
        ``errors`` lists the messages of the error-level issues.
    """
    report = validate_document(document)
    return ToolResult(
        ok=report.ok,
        payload=report.model_dump(mode="json"),
        errors=[issue.message for issue in report.errors],
    )


def render_tool(document: Mapping[str, Any]) -> bytes:
    """Re-render a candidate document as a panel image so the critic can compare it.

    The implementation will parse ``document``, draw it in the Stelltisch tile
    style through :mod:`rail_vision_bench.synth.render` and return PNG bytes.

    Args:
        document: A raw ``SceneAnnotation`` JSON object.

    Returns:
        The rendered image.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.tools.tools.render_tool is not implemented in the skeleton: "
        "render the document as a panel image"
    )
