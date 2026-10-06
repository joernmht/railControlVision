"""The single-shot baseline: one prompt, one image, one JSON document.

Large photos are downscaled before sending (every vision API downsamples anyway, and a
known scale lets the answer's pixel geometry be mapped back exactly); the prompt states the
size of the image the model actually sees, and :func:`rescale_geometry` maps the answer to
the original pixel grid that ``source.width`` / ``source.height`` describe.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PIL import Image
from pydantic import ValidationError

from rail_vision_bench.agents.prompts import load_prompt
from rail_vision_bench.providers.base import (
    ImageInput,
    VisionProvider,
    VisionRequest,
    VisionResponse,
)
from rail_vision_bench.schema.models import SceneAnnotation

MAX_SIDE: Final = 2048
"""Longest side of the image sent to a model, in pixels."""

MAX_OUTPUT_TOKENS: Final = 64000
"""Room for reasoning tokens plus a complete document of a dense panel (32000 truncated
some pre-fill answers; 64000 is the smallest output limit in the pre-fill lineup)."""

_GEOMETRY_KEYS: Final = frozenset({"bbox", "polyline", "point"})
_FENCE_RE: Final = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class SceneInput:
    """What the single-shot call needs to know about one scene."""

    scene_id: str
    source_kind: str
    image: Path
    width: int
    height: int


@dataclass(frozen=True)
class PreparedImage:
    """The image as sent and the factor that maps its pixels back to the original."""

    image: ImageInput
    width: int
    height: int
    scale: float


@dataclass
class SingleShotResult:
    """One attempt: the raw response, the parsed document or why parsing failed."""

    response: VisionResponse
    annotation: SceneAnnotation | None
    parse_error: str | None


def split_prompt(text: str) -> tuple[str, str]:
    """Split a packaged prompt into its ``## System`` and ``## User`` sections.

    Args:
        text: The Markdown prompt.

    Returns:
        The system text and the user template, both stripped.

    Raises:
        ValueError: If either section is missing.
    """
    system_at = text.find("\n## System\n")
    user_at = text.find("\n## User\n")
    if system_at < 0 or user_at < system_at:
        msg = "prompt needs a '## System' section followed by a '## User' section"
        raise ValueError(msg)
    system = text[system_at + len("\n## System\n") : user_at]
    return system.strip(), text[user_at + len("\n## User\n") :].strip()


def prepare_image(path: Path, *, max_side: int = MAX_SIDE) -> PreparedImage:
    """Load an image, downscale it to ``max_side`` and encode it as JPEG.

    Args:
        path: The source image.
        max_side: Longest side of the encoded image.

    Returns:
        The encoded image, its size and the factor original/sent (1.0 when not scaled).
    """
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    original = max(image.size)
    if original > max_side:
        factor = max_side / original
        image = image.resize(
            (round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return PreparedImage(
        image=ImageInput(data=buffer.getvalue(), media_type="image/jpeg"),
        width=image.width,
        height=image.height,
        scale=original / max(image.size),
    )


def extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model answer.

    Accepts a bare object, one wrapped in Markdown fences, or one surrounded by prose
    (the outermost ``{ ... }`` span is taken).

    Args:
        text: The model output.

    Returns:
        The decoded object.

    Raises:
        ValueError: If no JSON object can be decoded.
    """
    stripped = _FENCE_RE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        msg = "no JSON object in the answer"
        raise ValueError(msg)
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg = "the answer is JSON but not an object"
        raise ValueError(msg)
    return value


def rescale_geometry(value: Any, scale: float) -> Any:
    """Multiply every coordinate under a ``bbox``, ``polyline`` or ``point`` key by ``scale``.

    Malformed geometry is left as it is, so the schema pass reports it.

    Args:
        value: A decoded document or any part of one.
        scale: Factor from the sent image to the original image.

    Returns:
        A copy with scaled coordinates.
    """
    if isinstance(value, dict):
        return {
            key: _scale_coords(item, scale)
            if key in _GEOMETRY_KEYS
            else rescale_geometry(item, scale)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rescale_geometry(item, scale) for item in value]
    return value


def _scale_coords(value: Any, scale: float) -> Any:
    """Scale a (nested) list of numbers; anything else is returned unchanged."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(value * scale, 1)
    if isinstance(value, list):
        return [_scale_coords(item, scale) for item in value]
    return value


def build_request(
    model_id: str,
    scene: SceneInput,
    *,
    prompt: str | None = None,
    max_side: int = MAX_SIDE,
    extra: dict[str, Any] | None = None,
) -> tuple[VisionRequest, PreparedImage]:
    """Build the single-shot request for one scene.

    Args:
        model_id: The provider's model id.
        scene: The scene to transcribe.
        prompt: A prompt overriding the packaged ``single_shot`` prompt.
        max_side: Longest side of the image sent.
        extra: Provider-specific body fields (``VisionRequest.extra``).

    Returns:
        The request and the prepared image (for the scale).
    """
    system, user = split_prompt(prompt if prompt is not None else load_prompt("single_shot"))
    prepared = prepare_image(scene.image, max_side=max_side)
    text = (
        user.replace("{scene_id}", scene.scene_id)
        .replace("{source_kind}", scene.source_kind)
        .replace("{image}", scene.image.name)
        .replace("{width}", str(prepared.width))
        .replace("{height}", str(prepared.height))
    )
    request = VisionRequest(
        model=model_id,
        prompt=text,
        images=[prepared.image],
        system=system,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        extra=extra or {},
    )
    return request, prepared


def to_annotation(
    response_text: str, scene: SceneInput, prepared: PreparedImage, *, model_id: str, run_id: str
) -> SceneAnnotation:
    """Parse an answer into a document in the original pixel grid.

    The fields the caller knows better than the model (``scene_id``, ``source``,
    ``provenance``) are overwritten; everything else is the model's.

    Raises:
        ValueError: If the answer holds no JSON object or it fails the pydantic models.
    """
    document = rescale_geometry(extract_json(response_text), prepared.scale)
    document["schema_version"] = "v0"
    document["scene_id"] = scene.scene_id
    document["source"] = {
        "kind": scene.source_kind,
        "image": scene.image.as_posix(),
        "width": scene.width,
        "height": scene.height,
    }
    document["provenance"] = {"kind": "prediction", "model_id": model_id, "run_id": run_id}
    try:
        return SceneAnnotation.model_validate(document)
    except ValidationError as exc:
        msg = f"schema: {exc.error_count()} error(s); first: {exc.errors()[0]['msg']} at " + (
            "/".join(str(part) for part in exc.errors()[0]["loc"])
        )
        raise ValueError(msg) from exc


async def run_single_shot(
    provider: VisionProvider,
    model_id: str,
    scene: SceneInput,
    *,
    run_id: str,
    prompt: str | None = None,
    extra: dict[str, Any] | None = None,
) -> SingleShotResult:
    """Ask the model once for a complete schema v0 document.

    Validation (semantic rules) is left to the caller so that every attempt is recorded;
    a provider error propagates.

    Args:
        provider: The model backend.
        model_id: The provider's model id.
        scene: The scene to transcribe.
        run_id: Recorded in the document's provenance.
        prompt: A prompt overriding the packaged baseline.
        extra: Provider-specific body fields.

    Returns:
        The response with the parsed document, or the reason it could not be parsed.
    """
    request, prepared = build_request(model_id, scene, prompt=prompt, extra=extra)
    response = await provider.complete(request)
    try:
        annotation = to_annotation(response.text, scene, prepared, model_id=model_id, run_id=run_id)
    except ValueError as exc:
        return SingleShotResult(response=response, annotation=None, parse_error=str(exc))
    return SingleShotResult(response=response, annotation=annotation, parse_error=None)
