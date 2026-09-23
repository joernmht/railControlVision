"""Mistral (Pixtral) provider via the mistralai SDK."""

from __future__ import annotations

from typing import ClassVar

from mistralai.client import Mistral, errors as mistral_errors, models as mistral_models

from rail_vision_bench.providers._common import (
    call_with_retry,
    data_url,
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
    """Retry rate limits, 5xx, missing responses and connection failures."""
    if isinstance(exc, mistral_errors.MistralError):
        return is_transient_status(exc.status_code)
    if isinstance(exc, mistral_errors.NoResponseError):
        return True
    return is_transport_error(exc)


def _message_text(content: object) -> str:
    """Flatten an assistant message's content (a string or a list of chunks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            chunk.text for chunk in content if isinstance(chunk, mistral_models.TextChunk)
        )
    return ""


class MistralProvider:
    """Pixtral vision models through ``mistralai`` (the typed ``mistralai.client`` package).

    ``Mistral.chat.complete_async`` receives one ``image_url`` chunk per image
    (a base64 data URL) followed by the prompt text, with ``request.system`` as
    a system message. A ``response_schema`` switches on JSON mode
    (``response_format={"type": "json_object"}``) and, since JSON mode carries
    no schema, the schema is appended to the prompt; the answer is parsed with
    :func:`extract_json`. The client is created on first use from
    ``settings.mistral_api_key``.
    """

    name: ClassVar[str] = "mistral"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Mistral | None = None

    def _get_client(self) -> Mistral:
        """Create the client on first use (constructing the provider needs no key)."""
        if self._client is None:
            key = require_secret(self.settings.mistral_api_key, "MISTRAL_API_KEY", self.name)
            self._client = Mistral(api_key=key)
        return self._client

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one Pixtral chat completion.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The first choice's text, the JSON recovered from it, token usage and
            the latency of the successful attempt.

        Raises:
            ValueError: If ``MISTRAL_API_KEY`` is not configured.
            mistralai.client.errors.MistralError: If the call fails permanently or
                retries run out.
        """
        client = self._get_client()
        chunks: list[mistral_models.ContentChunkTypedDict] = [
            {"type": "image_url", "image_url": data_url(image)} for image in request.images
        ]
        chunks.append(
            {
                "type": "text",
                "text": with_schema_instruction(request.prompt, request.response_schema),
            }
        )
        messages: list[mistral_models.ChatCompletionRequestMessageTypedDict] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        user: mistral_models.UserMessageTypedDict = {"role": "user", "content": chunks}
        messages.append(user)
        response_format: mistral_models.ResponseFormatTypedDict | None = (
            {"type": "json_object"} if request.response_schema is not None else None
        )

        async def call() -> mistral_models.ChatCompletionResponse:
            return await client.chat.complete_async(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=response_format,
            )

        response, latency_ms = await call_with_retry(call, _is_transient)
        message = response.choices[0].message if response.choices else None
        text = _message_text(message.content) if message is not None else ""
        return VisionResponse(
            text=text,
            parsed=parse_answer(text),
            usage=Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            ),
            latency_ms=latency_ms,
            raw=dump_raw(response),
        )
