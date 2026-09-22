"""Google Gemini provider via the google-genai SDK, stubbed in the skeleton."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class GeminiProvider:
    """Gemini multimodal models through ``google-genai``.

    The implementation will build a ``genai.Client`` from
    ``settings.google_api_key``, pass the images as inline ``Part`` blobs
    with their media type, and ask for ``application/json`` output constrained
    by ``response_schema``.
    """

    name: ClassVar[str] = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Gemini generate-content call.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.gemini.GeminiProvider.complete is not implemented "
            "in the skeleton: call Gemini generate_content with inline image parts"
        )
