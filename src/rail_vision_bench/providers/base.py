"""Provider-independent request and response models and the ``VisionProvider`` protocol.

Every provider turns one :class:`VisionRequest` (prompt, images, optional JSON
schema) into one :class:`VisionResponse`; the benchmark never touches an SDK
directly, so a new backend only has to implement :class:`VisionProvider`.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ImageInput(BaseModel):
    """One image sent to a model, as raw bytes plus its media type."""

    model_config = ConfigDict(extra="forbid")

    data: bytes
    media_type: Literal["image/png", "image/jpeg", "image/webp"] = "image/png"


class Usage(BaseModel):
    """Token counts and cost of one completion."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class VisionRequest(BaseModel):
    """What the benchmark asks a model to do with one or more images."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    images: list[ImageInput] = Field(default_factory=list)
    system: str | None = None
    response_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema the answer must conform to (schema v0 for scenes)."
    )
    max_tokens: int = 4096
    temperature: float = 0.0


class VisionResponse(BaseModel):
    """What a provider returns: the text, the parsed JSON when available, usage and latency."""

    model_config = ConfigDict(extra="forbid")

    text: str
    parsed: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    raw: Any = Field(default=None, description="The provider SDK's own response object, if kept.")


class VisionProvider(Protocol):
    """The contract every provider implements.

    ``name`` is the registry key; it is a ``ClassVar`` so that implementations
    can declare it as a class attribute (mypy rejects a class variable where a
    protocol asks for an instance variable).
    """

    name: ClassVar[str]

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one vision completion.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer.
        """
        ...
