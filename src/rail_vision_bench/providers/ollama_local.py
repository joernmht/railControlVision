"""Local models (Qwen-VL, LLaVA, ...) served by Ollama."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import ollama

from rail_vision_bench.providers._common import (
    call_with_retry,
    dump_raw,
    is_transient_status,
    is_transport_error,
    parse_answer,
)
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings


def _is_transient(exc: BaseException) -> bool:
    """Retry 429/5xx answers (model loading, overload) and connection failures."""
    if isinstance(exc, ollama.ResponseError):
        return is_transient_status(exc.status_code)
    return is_transport_error(exc)


class OllamaProvider:
    """Locally hosted vision models through the ``ollama`` client.

    An ``ollama.AsyncClient`` connected to ``settings.ollama_host`` is created
    on first use (no API key is involved). The user message carries the prompt
    text and the raw image bytes in its ``images`` list, ``request.system``
    becomes a system message, and ``max_tokens``/``temperature`` map to the
    ``num_predict``/``temperature`` options. A ``response_schema`` is passed as
    ``format`` so the server constrains decoding to it; the answer is still
    parsed with :func:`extract_json`.
    """

    name: ClassVar[str] = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: ollama.AsyncClient | None = None

    def _get_client(self) -> ollama.AsyncClient:
        """Create the async client on first use."""
        if self._client is None:
            self._client = ollama.AsyncClient(host=self.settings.ollama_host)
        return self._client

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one chat call against the local Ollama server.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The message text, the JSON recovered from it, token usage
            (``prompt_eval_count``/``eval_count``) and the latency of the
            successful attempt.

        Raises:
            ollama.ResponseError: If the server rejects the request permanently or
                retries run out.
            ConnectionError: If the server stays unreachable after the retries.
        """
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append(
            {
                "role": "user",
                "content": request.prompt,
                "images": [ollama.Image(value=image.data) for image in request.images],
            }
        )
        output_format: dict[str, Any] | Literal[""] = (
            request.response_schema if request.response_schema is not None else ""
        )

        async def call() -> ollama.ChatResponse:
            return await client.chat(
                model=request.model,
                messages=messages,
                format=output_format,
                options={"temperature": request.temperature, "num_predict": request.max_tokens},
            )

        response, latency_ms = await call_with_retry(call, _is_transient)
        text = response.message.content or ""
        return VisionResponse(
            text=text,
            parsed=parse_answer(text),
            usage=Usage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
            ),
            latency_ms=latency_ms,
            raw=dump_raw(response),
        )
