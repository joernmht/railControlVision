"""Anthropic Claude provider (vision plus tool use), stubbed in the skeleton."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class AnthropicProvider:
    """Claude via the Anthropic SDK.

    The implementation will send ``messages.create`` requests whose content
    holds one image block per :class:`ImageInput` followed by the prompt text,
    force structured output through tool use with ``response_schema`` as the
    tool input schema, and offer the Message Batches API for large offline
    runs. Credentials come from ``settings.anthropic_api_key``.
    """

    name: ClassVar[str] = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Claude vision completion.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.claude.AnthropicProvider.complete is not implemented "
            "in the skeleton: call the Anthropic messages API with image blocks and tool use"
        )
