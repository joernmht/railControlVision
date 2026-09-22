"""Mistral (Pixtral) provider via the mistralai SDK, stubbed in the skeleton."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class MistralProvider:
    """Pixtral vision models through ``mistralai``.

    The implementation will use the async chat completion of the Mistral SDK
    with ``settings.mistral_api_key``, encode the images as base64 data URLs in
    ``image_url`` chunks and request JSON mode; the SDK ships no type
    information, so it is imported only inside the future implementation.
    """

    name: ClassVar[str] = "mistral"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Pixtral chat completion.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.mistral.MistralProvider.complete is not implemented "
            "in the skeleton: call the Mistral chat API with image_url chunks"
        )
