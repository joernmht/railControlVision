"""OpenAI and OpenAI-compatible provider (vLLM, Ollama's OpenAI endpoint)."""

from __future__ import annotations

from typing import Any, ClassVar, Final

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    completion_create_params,
)
from openai.types.shared_params import ResponseFormatJSONSchema

from rail_vision_bench.providers._common import (
    call_with_retry,
    data_url,
    dump_raw,
    is_transient_status,
    is_transport_error,
    parse_answer,
    schema_name,
)
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings

LOCAL_API_KEY: Final[str] = "not-needed"
"""Placeholder key for self-hosted servers (vLLM, Ollama) that do not check one."""


def _is_transient(exc: BaseException) -> bool:
    """Retry rate limits, 5xx, timeouts and connection failures."""
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return is_transient_status(exc.status_code)
    return is_transport_error(exc)


def schema_response_format(
    schema: dict[str, Any],
) -> completion_create_params.ResponseFormat:
    """Build a non-strict ``json_schema`` response format for a JSON Schema.

    Strict mode is not used because schema v0 (``patternProperties``, optional
    fields, ``anyOf`` with defaults) does not satisfy OpenAI's strict subset.

    Args:
        schema: The response schema.

    Returns:
        The ``response_format`` argument, named by :func:`schema_name`.
    """
    response_format: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {"name": schema_name(schema), "schema": schema, "strict": False},
    }
    return response_format


class OpenAICompatProvider:
    """Any chat-completions server that speaks the OpenAI wire protocol.

    Uses ``openai.AsyncOpenAI`` with ``base_url`` taken from
    ``settings.openai_base_url`` (unset means api.openai.com; a vLLM or
    Ollama-compatible server otherwise). Images go as data URLs in
    ``image_url`` content parts before the prompt text; a ``response_schema``
    becomes a non-strict ``json_schema`` ``response_format`` and the answer
    text is parsed with :func:`extract_json`. Against api.openai.com the key
    ``OPENAI_API_KEY`` is required and ``max_completion_tokens`` is sent; a
    custom base URL works without a key and gets the older ``max_tokens``,
    which self-hosted servers understand. The client is created on first use
    with SDK-level retries disabled (tenacity handles them).
    """

    name: ClassVar[str] = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Create the async client on first use (constructing the provider needs no key)."""
        if self._client is None:
            secret = self.settings.openai_api_key
            key = secret.get_secret_value() if secret is not None else ""
            base_url = self.settings.openai_base_url or None
            if not key:
                if base_url is None:
                    msg = (
                        "the openai provider needs an API key: set OPENAI_API_KEY in the "
                        "environment or .env (or OPENAI_BASE_URL for a server without keys)"
                    )
                    raise ValueError(msg)
                key = LOCAL_API_KEY
            self._client = openai.AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)
        return self._client

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one chat completion against the configured base URL.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The first choice's text, the JSON recovered from it, token usage and
            the latency of the successful attempt.

        Raises:
            ValueError: If neither ``OPENAI_API_KEY`` nor ``OPENAI_BASE_URL`` is configured.
            openai.OpenAIError: If the API call fails permanently or retries run out.
        """
        client = self._get_client()
        parts: list[ChatCompletionContentPartParam] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in request.images
        ]
        parts.append({"type": "text", "text": request.prompt})
        messages: list[ChatCompletionMessageParam] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": parts})
        response_format: completion_create_params.ResponseFormat | openai.Omit = openai.omit
        if request.response_schema is not None:
            response_format = schema_response_format(request.response_schema)
        hosted = not self.settings.openai_base_url

        async def call() -> ChatCompletion:
            return await client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_completion_tokens=request.max_tokens if hosted else openai.omit,
                max_tokens=openai.omit if hosted else request.max_tokens,
                response_format=response_format,
            )

        completion, latency_ms = await call_with_retry(call, _is_transient)
        text = (completion.choices[0].message.content or "") if completion.choices else ""
        usage = completion.usage
        return VisionResponse(
            text=text,
            parsed=parse_answer(text),
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage is not None else 0,
                output_tokens=usage.completion_tokens if usage is not None else 0,
            ),
            latency_ms=latency_ms,
            raw=dump_raw(completion),
        )
