"""LiteLLM unified router over many hosted and local models, stubbed in the skeleton."""

from __future__ import annotations

from typing import ClassVar

from rail_vision_bench.providers.base import VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


class LiteLLMProvider:
    """One interface for every backend LiteLLM knows, with built-in cost tracking.

    The implementation will call ``litellm.acompletion`` with the model id in
    LiteLLM's ``provider/model`` notation, images as data-URL content parts,
    and fill ``Usage.cost_usd`` from ``litellm.completion_cost`` so cost
    metrics need no per-provider price tables. Keys are read from the same
    settings the dedicated providers use.
    """

    name: ClassVar[str] = "litellm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one completion through LiteLLM.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.

        Raises:
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError(
            "rail_vision_bench.providers.litellm_router.LiteLLMProvider.complete is not "
            "implemented in the skeleton: route the request through litellm.acompletion"
        )
