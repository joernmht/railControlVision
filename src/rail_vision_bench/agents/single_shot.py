"""The single-shot baseline: one prompt, one image, one JSON document."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.agents.documents import (
    default_source,
    parse_scene,
    scene_id_for,
    stamp_document,
)
from rail_vision_bench.agents.prompts import fill_prompt, load_prompt, split_prompt
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider, VisionRequest
from rail_vision_bench.providers.parsing import extract_json
from rail_vision_bench.schema.export import to_json_schema
from rail_vision_bench.schema.models import SceneAnnotation, Source


class ParseError(ValueError):
    """The model answer holds no schema v0 document.

    Raised by :func:`run_single_shot`; ``raw_text`` is the model output so the
    caller can record the failed attempt, ``document`` the recovered JSON
    object when there was one (it then failed the schema).
    """

    def __init__(
        self, message: str, *, raw_text: str, document: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.document = document


class SingleShotResult(BaseModel):
    """Everything one single-shot attempt produced, parsed or not."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    annotation: SceneAnnotation | None = Field(
        default=None, description="The parsed document, None when parsing failed."
    )
    document: dict[str, Any] | None = Field(
        default=None, description="The stamped raw JSON object, also when it fails the schema."
    )
    raw_text: str = Field(description="The model output before parsing.")
    parse_error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(description="Wall-clock time of the provider call.")


def build_single_shot_request(
    image: ImageInput, *, model: str, scene_id: str, source: Source, prompt: str | None = None
) -> VisionRequest:
    """Build the request of one single-shot attempt.

    The packaged ``single_shot`` prompt is split into its system and user
    turns and its ``{scene_id}``, ``{source_kind}``, ``{image}``, ``{width}``
    and ``{height}`` placeholders are filled; a custom ``prompt`` goes through
    the same treatment. The JSON Schema of schema v0 is the ``response_schema``.

    Args:
        image: The panel or screen image.
        model: The SDK model id.
        scene_id: The scene id the document must carry.
        source: The source block of the scene.
        prompt: A prompt overriding the packaged baseline.

    Returns:
        The request.
    """
    system, user = split_prompt(prompt if prompt is not None else load_prompt("single_shot"))
    values = {
        "scene_id": scene_id,
        "source_kind": source.kind.value,
        "image": source.image,
        "width": source.width,
        "height": source.height,
    }
    if not user:
        # A prompt without sections is a plain instruction: send it as the user turn.
        system, user = "", system
    return VisionRequest(
        model=model,
        system=fill_prompt(system, **values) or None,
        prompt=fill_prompt(user, **values),
        images=[image],
        response_schema=to_json_schema(),
    )


async def single_shot_attempt(
    provider: VisionProvider,
    image: ImageInput,
    *,
    model: str,
    prompt: str | None = None,
    scene_id: str | None = None,
    source: Source | None = None,
) -> SingleShotResult:
    """Run one single-shot attempt and report its outcome without raising on bad output.

    The provider's ``parsed`` JSON is preferred; otherwise the object is
    recovered from the text with :func:`~rail_vision_bench.providers.parsing.extract_json`.
    ``schema_version``, ``scene_id``, ``source`` and ``provenance`` are then
    overwritten by code before the document is parsed. Validation against the
    semantic rules is left to the caller so that every attempt is recorded.

    Args:
        provider: The model backend.
        image: The panel or screen image.
        model: The SDK model id sent with the request.
        prompt: A prompt overriding the packaged baseline.
        scene_id: The scene id; derived from the image bytes when omitted.
        source: The source block (manifest data); derived from the image when omitted.

    Returns:
        The attempt: annotation or parse error, raw text, usage and latency.

    Raises:
        ValueError: If ``source`` is omitted and the image cannot be decoded.
    """
    scene = scene_id if scene_id is not None else scene_id_for(image)
    src = source if source is not None else default_source(image, scene_id=scene)
    request = build_single_shot_request(
        image, model=model, scene_id=scene, source=src, prompt=prompt
    )
    started = time.perf_counter()
    response = await provider.complete(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw_text = response.text
    if not raw_text and response.parsed is not None:
        raw_text = json.dumps(response.parsed, ensure_ascii=False)
    recovered = response.parsed if response.parsed is not None else extract_json(response.text)
    if recovered is None:
        return SingleShotResult(
            scene_id=scene,
            raw_text=raw_text,
            parse_error="the model output contains no JSON object",
            usage=response.usage,
            latency_ms=latency_ms,
        )
    document = stamp_document(recovered, scene_id=scene, source=src, model=model)
    annotation, error = parse_scene(document)
    return SingleShotResult(
        scene_id=scene,
        annotation=annotation,
        document=document,
        raw_text=raw_text,
        parse_error=error,
        usage=response.usage,
        latency_ms=latency_ms,
    )


async def run_single_shot(
    provider: VisionProvider,
    image: ImageInput,
    *,
    model: str,
    prompt: str | None = None,
    scene_id: str | None = None,
    source: Source | None = None,
) -> SceneAnnotation:
    """Ask the model once for a complete schema v0 document.

    A thin wrapper over :func:`single_shot_attempt` for callers that only want
    the document.

    Args:
        provider: The model backend.
        image: The panel or screen image.
        model: The SDK model id sent with the request.
        prompt: A prompt overriding the packaged baseline.
        scene_id: The scene id; derived from the image bytes when omitted.
        source: The source block; derived from the image when omitted.

    Returns:
        The parsed document (not yet checked against the semantic rules).

    Raises:
        ParseError: If the answer holds no JSON object or the object fails schema v0.
        ValueError: If ``source`` is omitted and the image cannot be decoded.
    """
    result = await single_shot_attempt(
        provider, image, model=model, prompt=prompt, scene_id=scene_id, source=source
    )
    if result.annotation is None:
        raise ParseError(
            result.parse_error or "the model output could not be parsed",
            raw_text=result.raw_text,
            document=result.document,
        )
    return result.annotation
