"""Google Gemini provider via the google-genai SDK."""

from __future__ import annotations

from typing import ClassVar

from google import genai
from google.genai import errors as genai_errors, types as genai_types

from rail_vision_bench.providers._common import (
    call_with_retry,
    dump_raw,
    is_transient_status,
    is_transport_error,
    parse_answer,
    require_secret,
    with_schema_instruction,
)
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


def _is_transient(exc: BaseException) -> bool:
    """Retry 429 (resource exhausted), 5xx and connection failures."""
    if isinstance(exc, genai_errors.APIError):
        return is_transient_status(exc.code)
    return is_transport_error(exc)


class GeminiProvider:
    """Gemini multimodal models through ``google-genai``.

    A ``genai.Client`` is built on first use from ``settings.google_api_key``
    and its async surface (``client.aio.models.generate_content``) receives
    one inline ``Part`` blob per image, with its media type, followed by the
    prompt text; ``request.system`` becomes the ``system_instruction``.

    With a ``response_schema`` the request asks for ``application/json`` output
    and appends the schema to the prompt instead of passing it as
    ``response_schema``/``response_json_schema``: Gemini accepts only a subset
    of JSON Schema and schema v0 uses keywords outside it (``pattern``,
    ``patternProperties``, ``propertyNames``, ``const``), so constraining
    decoding with it would be rejected or silently lossy. The answer is parsed
    with :func:`extract_json`.
    """

    name: ClassVar[str] = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Create the client on first use (constructing the provider needs no key)."""
        if self._client is None:
            key = require_secret(self.settings.google_api_key, "GOOGLE_API_KEY", self.name)
            self._client = genai.Client(api_key=key)
        return self._client

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Gemini generate-content call.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The response text, the JSON recovered from it, token usage (output
            counts candidate plus thinking tokens) and the latency of the
            successful attempt.

        Raises:
            ValueError: If ``GOOGLE_API_KEY`` is not configured.
            google.genai.errors.APIError: If the call fails permanently or retries run out.
        """
        client = self._get_client()
        parts: list[genai_types.Part] = [
            genai_types.Part.from_bytes(data=image.data, mime_type=image.media_type)
            for image in request.images
        ]
        parts.append(
            genai_types.Part.from_text(
                text=with_schema_instruction(request.prompt, request.response_schema)
            )
        )
        contents = genai_types.Content(role="user", parts=parts)
        config = genai_types.GenerateContentConfig(
            system_instruction=request.system,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            response_mime_type="application/json" if request.response_schema is not None else None,
            # no tools are sent; disabling AFC also skips the SDK's AFC code path and warning
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        async def call() -> genai_types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=request.model, contents=contents, config=config
            )

        response, latency_ms = await call_with_retry(call, _is_transient)
        text = _response_text(response)
        meta = response.usage_metadata
        input_tokens = (meta.prompt_token_count or 0) if meta is not None else 0
        output_tokens = (
            (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
            if meta is not None
            else 0
        )
        return VisionResponse(
            text=text,
            parsed=parse_answer(text),
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            latency_ms=latency_ms,
            raw=dump_raw(response),
        )


def _response_text(response: genai_types.GenerateContentResponse) -> str:
    """Concatenate the non-thought text parts of the first candidate."""
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if content is None or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text and not part.thought)
