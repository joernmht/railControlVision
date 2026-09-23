"""Helpers shared by the single-shot runner and the agentic loop.

The model is trusted with the content of a scene, never with its bookkeeping:
``schema_version``, ``scene_id``, ``source`` and ``provenance`` are always
written by code (:func:`stamp_document`), so a prediction can be joined with
its ground truth and its model no matter what the model echoed back.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from rail_vision_bench import SCHEMA_VERSION
from rail_vision_bench.providers.base import ImageInput, Usage
from rail_vision_bench.schema.models import ProvenanceKind, SceneAnnotation, Source, SourceKind

_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def scene_id_for(image: ImageInput) -> str:
    """Derive a stable scene id from the image content.

    Args:
        image: The image the scene describes.

    Returns:
        ``frame-<first 12 hex digits of the SHA-256 of the bytes>``.
    """
    return f"frame-{hashlib.sha256(image.data).hexdigest()[:12]}"


def image_size(image: ImageInput) -> tuple[int, int]:
    """Decode the image header and return its pixel size.

    Args:
        image: The encoded image.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If the bytes are not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            return decoded.size
    except (UnidentifiedImageError, OSError) as exc:
        msg = f"cannot decode the {image.media_type} image: {exc}"
        raise ValueError(msg) from exc


def default_source(
    image: ImageInput, *, scene_id: str, kind: SourceKind = SourceKind.STREAM_FRAME
) -> Source:
    """Describe an image that has no manifest row (a harness frame, an ad-hoc call).

    Args:
        image: The encoded image; its size is read from the header.
        scene_id: The scene id, used as the image name.
        kind: Where the image comes from.

    Returns:
        The ``source`` block of the prediction.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    width, height = image_size(image)
    return Source(
        kind=kind,
        image=f"{scene_id}.{_EXTENSIONS[image.media_type]}",
        width=width,
        height=height,
    )


def stamp_document(
    document: Mapping[str, Any], *, scene_id: str, source: Source, model: str
) -> dict[str, Any]:
    """Overwrite the bookkeeping fields of a model-produced document.

    Args:
        document: The raw document as the model produced it.
        scene_id: The scene id the caller assigned.
        source: The source block the caller knows to be true.
        model: The SDK model id that produced the document.

    Returns:
        A copy with ``schema_version``, ``scene_id``, ``source`` and
        ``provenance`` (kind ``prediction``, ``model_id``) set by code.
    """
    stamped = dict(document)
    stamped["schema_version"] = SCHEMA_VERSION
    stamped["scene_id"] = scene_id
    stamped["source"] = source.model_dump(mode="json", exclude_none=True)
    stamped["provenance"] = {"kind": ProvenanceKind.PREDICTION.value, "model_id": model}
    return stamped


def parse_scene(document: Mapping[str, Any]) -> tuple[SceneAnnotation | None, str | None]:
    """Parse a raw document into the schema v0 model without raising.

    Args:
        document: The raw document.

    Returns:
        ``(annotation, None)`` on success, ``(None, message)`` when pydantic
        rejects the document; the message summarises the first errors.
    """
    try:
        return SceneAnnotation.model_validate(dict(document)), None
    except ValidationError as exc:
        errors = exc.errors()
        shown = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in errors[:5]
        )
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        return None, f"document does not match schema v0: {shown}{more}"


def add_usage(total: Usage, extra: Usage) -> Usage:
    """Sum two usage records; the cost stays None only when both are unknown.

    Args:
        total: The running total.
        extra: The usage of one more call.

    Returns:
        A new record with the summed token counts and cost.
    """
    if total.cost_usd is None and extra.cost_usd is None:
        cost = None
    else:
        cost = (total.cost_usd or 0.0) + (extra.cost_usd or 0.0)
    return Usage(
        input_tokens=total.input_tokens + extra.input_tokens,
        output_tokens=total.output_tokens + extra.output_tokens,
        cost_usd=cost,
    )
