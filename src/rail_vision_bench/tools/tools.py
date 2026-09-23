"""The four agent tools: crop, read, validate and render.

``crop_tool`` cuts and enlarges a region with Pillow, ``read_tool`` asks a
vision provider one focused question about a crop and parses its JSON answer,
``validate_tool`` wraps the validator and ``render_tool`` draws a candidate
document as a panel image through :mod:`rail_vision_bench.synth.render`.
Heavy imports happen inside the tools so importing this module stays cheap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.schema.models import SceneAnnotation

if TYPE_CHECKING:
    from rail_vision_bench.providers.base import VisionProvider

MediaType = Literal["image/png", "image/jpeg", "image/webp"]

TOOL_NAMES: Final[tuple[str, ...]] = ("crop", "read", "validate", "render")
"""Tool names as exposed to agents and over MCP."""

READ_PROMPT: Final[str] = (
    "You are reading a crop of a railway control panel or interlocking screen.\n"
    "Question: {question}\n"
    "Answer only with a JSON object of the form "
    '{{"text": "<your answer>", "confidence": <number between 0 and 1>}}. '
    "Transcribe labels exactly as shown; if the crop does not show the answer, "
    'use an empty "text" and a low confidence.'
)
"""Prompt of :func:`read_tool`; ``{question}`` is replaced by the question."""

READ_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["text", "confidence"],
    "additionalProperties": False,
}
"""JSON Schema of the answer :func:`read_tool` asks for."""


class ToolResult(BaseModel):
    """Uniform tool outcome: success flag, structured payload and human-readable errors."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def crop_tool(image: bytes, bbox: tuple[float, float, float, float], *, zoom: float = 2.0) -> bytes:
    """Cut a region out of an image and upscale it for reading small labels.

    ``image`` is decoded with Pillow (EXIF orientation applied, converted to
    RGB), ``bbox`` (``x0, y0, x1, y1`` in pixels) is clipped to the image and
    rounded outwards to whole pixels, and the crop is resized by ``zoom`` with
    Lanczos resampling.

    Args:
        image: Encoded image bytes (any format Pillow reads).
        bbox: The region in pixel coordinates.
        zoom: Upscaling factor applied to the crop.

    Returns:
        The crop encoded as PNG.

    Raises:
        ValueError: If ``zoom`` is not positive, the image cannot be decoded,
            or ``bbox`` has no area inside the image.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    from rail_vision_bench.ingest.preprocess import clip_bbox

    if zoom <= 0:
        msg = f"zoom must be positive, got {zoom}"
        raise ValueError(msg)
    try:
        with Image.open(BytesIO(image)) as opened:
            decoded = ImageOps.exif_transpose(opened).convert("RGB")
    except UnidentifiedImageError as exc:
        msg = "image bytes are not a decodable image"
        raise ValueError(msg) from exc
    x0, y0, x1, y1 = clip_bbox(bbox, decoded.width, decoded.height)
    crop = decoded.crop((x0, y0, x1, y1))
    size = (max(1, round(crop.width * zoom)), max(1, round(crop.height * zoom)))
    enlarged = crop.resize(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    enlarged.save(buffer, format="PNG")
    return buffer.getvalue()


def _media_type(image: bytes) -> tuple[bytes, MediaType]:
    """Detect PNG, JPEG or WebP from the magic bytes; re-encode anything else as PNG."""
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return image, "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return image, "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return image, "image/webp"
    from PIL import Image

    with Image.open(BytesIO(image)) as opened:
        buffer = BytesIO()
        opened.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


async def read_tool(
    image: bytes, question: str, *, provider: VisionProvider, model: str
) -> ToolResult:
    """Ask a focused question about an image crop (a label, a lamp, an indication).

    The image is sent with its detected media type (PNG, JPEG or WebP; other
    formats Pillow reads are re-encoded as PNG) and a prompt that asks
    ``question`` and requests the JSON object ``{"text": ..., "confidence":
    0..1}``. The answer is parsed with
    :func:`rail_vision_bench.providers.parsing.extract_json` (the provider's
    own ``parsed`` object is preferred when present); when it holds no
    ``text``, the raw answer is the text and the confidence is ``None``. A
    confidence outside ``[0, 1]`` or of the wrong type is reported as ``None``.

    Args:
        image: Encoded crop bytes.
        question: What to read or decide, e.g. "Which digits follow 'W'?".
        provider: The vision backend to ask.
        model: Model id passed to the provider.

    Returns:
        ``ok=True`` with ``payload`` ``{"text": str, "confidence": float |
        None, "usage": {"input_tokens", "output_tokens", "cost_usd"}}``, or
        ``ok=False`` with an error when the image cannot be decoded or the
        question is empty.

    Raises:
        Exception: Whatever ``provider.complete`` raises (network, auth,
            ``NotImplementedError``) propagates unchanged.
    """
    from rail_vision_bench.providers.base import ImageInput, VisionRequest
    from rail_vision_bench.providers.parsing import extract_json

    if not question.strip():
        return ToolResult(ok=False, errors=["question must not be empty"])
    try:
        data, media_type = _media_type(image)
    except (OSError, ValueError) as exc:
        return ToolResult(ok=False, errors=[f"image cannot be decoded: {exc}"])
    request = VisionRequest(
        model=model,
        prompt=READ_PROMPT.format(question=question.strip()),
        images=[ImageInput(data=data, media_type=media_type)],
        response_schema=READ_RESPONSE_SCHEMA,
        max_tokens=512,
        temperature=0.0,
    )
    response = await provider.complete(request)
    parsed = response.parsed if response.parsed is not None else extract_json(response.text)
    text: str = response.text.strip()
    confidence: float | None = None
    if parsed is not None and parsed.get("text") is not None:
        value = parsed["text"]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        confidence = _confidence(parsed.get("confidence"))
    return ToolResult(
        ok=True,
        payload={
            "text": text,
            "confidence": confidence,
            "usage": response.usage.model_dump(mode="json"),
        },
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


def render_tool(document: Mapping[str, Any], *, style: str = "stelltisch") -> bytes:
    """Re-render a candidate document as a panel image so the critic can compare it.

    ``document`` is parsed as a ``SceneAnnotation`` (structure only; semantic
    errors are drawn as they are) and drawn through
    :func:`rail_vision_bench.synth.render.render_png` at its source size: from
    its own geometry when every node and edge has some, otherwise from an
    automatic tile layout.

    Args:
        document: A raw ``SceneAnnotation`` JSON object.
        style: Render style, ``stelltisch`` (default) or ``estw``.

    Returns:
        The rendered image as PNG bytes.

    Raises:
        pydantic.ValidationError: If ``document`` is not a structurally valid
            ``SceneAnnotation`` (a ``ValueError`` subclass).
        ValueError: If ``style`` is unknown.
    """
    from rail_vision_bench.synth.render import render_png

    return render_png(SceneAnnotation.model_validate(document), style=style)
