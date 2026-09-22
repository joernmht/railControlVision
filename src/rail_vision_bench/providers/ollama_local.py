"""Local models (Qwen-VL, LLaVA, ...) served by Ollama, stubbed in the skeleton."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class OllamaProvider:
    """Locally hosted vision models through the ``ollama`` client.

    The implementation will connect an ``ollama.AsyncClient`` to
    ``settings.ollama_host``, pass the raw image bytes in the message's
    ``images`` list and use ``format`` with ``response_schema`` for
    constrained JSON output. No API key is involved.
    """

    name: ClassVar[str] = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one chat call against the local Ollama server.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.ollama_local.OllamaProvider.complete is not implemented "
            "in the skeleton: call the Ollama chat endpoint with raw image bytes"
        )
