"""The single-shot baseline: one prompt, one image, one JSON document (stub)."""

from __future__ import annotations

from rail_vision_bench.providers.base import ImageInput, VisionProvider
from rail_vision_bench.schema.models import SceneAnnotation


async def run_single_shot(
    provider: VisionProvider, image: ImageInput, *, prompt: str | None = None
) -> SceneAnnotation:
    """Ask the model once for a complete schema v0 document.

    The implementation will load ``prompts/single_shot.md`` when ``prompt`` is
    omitted, send it with the image and the JSON Schema of ``SceneAnnotation``
    as ``response_schema``, parse the answer and return the model instance;
    validation is left to the caller so that every attempt is recorded.

    Args:
        provider: The model backend.
        image: The panel or screen image.
        prompt: A prompt overriding the packaged baseline.

    Returns:
        The parsed document.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.single_shot.run_single_shot is not implemented in the "
        "skeleton: send the single-shot prompt with the image and parse one JSON document"
    )
