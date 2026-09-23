"""Anthropic Claude provider (vision plus tool use for structured output)."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Final

import anthropic
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
)

from rail_vision_bench.providers._common import (
    b64,
    call_with_retry,
    dump_raw,
    is_transient_status,
    is_transport_error,
    parse_answer,
    require_secret,
)
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings

RESULT_TOOL: Final[str] = "emit_result"
"""Name of the tool whose forced call carries the structured answer."""


def _is_transient(exc: BaseException) -> bool:
    """Retry rate limits, overload (529), 5xx and connection failures."""
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return is_transient_status(exc.status_code)
    return is_transport_error(exc)


class AnthropicProvider:
    """Claude via the Anthropic SDK's ``AsyncAnthropic`` client.

    ``messages.create`` receives one base64 image block per :class:`ImageInput`
    followed by the prompt text. With a ``response_schema`` the request carries
    a single tool (``emit_result``) whose ``input_schema`` is the schema and
    ``tool_choice`` forces that tool, so the parsed answer is the tool input;
    the text blocks still go through :func:`extract_json` as a fallback. The
    client is created on the first call from ``settings.anthropic_api_key``
    with SDK-level retries disabled (tenacity handles them).
    ``request.temperature`` is not sent: the installed SDK's ``messages.create``
    no longer takes sampling parameters for current Claude models. The
    Message Batches API for large offline runs is not wired up yet.
    """

    name: ClassVar[str] = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Create the async client on first use (constructing the provider needs no key)."""
        if self._client is None:
            key = require_secret(self.settings.anthropic_api_key, "ANTHROPIC_API_KEY", self.name)
            self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=0)
        return self._client

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Claude vision completion.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The concatenated text blocks (the JSON dump of the tool input when the
            model wrote no text), the parsed object (forced tool input
            when a schema was given, else JSON recovered from the text), token
            usage and the latency of the successful attempt.

        Raises:
            ValueError: If ``ANTHROPIC_API_KEY`` is not configured.
            anthropic.APIError: If the API call fails permanently or retries run out.
        """
        client = self._get_client()
        content: list[ImageBlockParam | TextBlockParam] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": image.media_type, "data": b64(image)},
            }
            for image in request.images
        ]
        content.append({"type": "text", "text": request.prompt})
        messages: list[MessageParam] = [{"role": "user", "content": content}]
        tools: list[ToolParam] | anthropic.Omit = anthropic.omit
        tool_choice: ToolChoiceToolParam | anthropic.Omit = anthropic.omit
        if request.response_schema is not None:
            tools = [
                {
                    "name": RESULT_TOOL,
                    "description": "Return the answer as one JSON object matching the schema.",
                    "input_schema": request.response_schema,
                }
            ]
            tool_choice = {"type": "tool", "name": RESULT_TOOL}

        async def call() -> anthropic.types.Message:
            return await client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system if request.system is not None else anthropic.omit,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

        message, latency_ms = await call_with_retry(call, _is_transient)
        text = "".join(block.text for block in message.content if block.type == "text")
        structured: dict[str, Any] | None = None
        for block in message.content:
            if block.type == "tool_use" and block.name == RESULT_TOOL:
                if isinstance(block.input, dict):
                    structured = block.input
                break
        if not text and structured is not None:
            text = json.dumps(structured, ensure_ascii=False)
        return VisionResponse(
            text=text,
            parsed=parse_answer(text, structured),
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            latency_ms=latency_ms,
            raw=dump_raw(message),
        )
