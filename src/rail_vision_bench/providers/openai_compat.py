"""OpenAI and OpenAI-compatible provider (vLLM, Ollama's OpenAI endpoint), stubbed."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class OpenAICompatProvider:
    """Any chat-completions server that speaks the OpenAI wire protocol.

    The implementation will use the OpenAI SDK's async client with
    ``base_url`` taken from ``settings.openai_base_url`` (unset means
    api.openai.com; a vLLM or Ollama-compatible server otherwise), send the
    images as data URLs in ``image_url`` content parts and request JSON output
    through ``response_format`` with ``response_schema`` when the server
    supports structured outputs.
    """

    name: ClassVar[str] = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one chat completion against the configured base URL.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.openai_compat.OpenAICompatProvider.complete is not "
            "implemented in the skeleton: call an OpenAI-compatible chat completions endpoint"
        )
