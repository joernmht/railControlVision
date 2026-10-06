"""OpenAI and OpenAI-compatible provider (OpenRouter, vLLM, Ollama's OpenAI endpoint).

Spoken over plain ``httpx`` rather than the OpenAI SDK: the wire format is small, the
async client is already a core dependency, and OpenRouter-specific fields (usage
accounting with the billed cost) need no SDK escape hatches.
"""

from __future__ import annotations

import base64
import time
from typing import Any, ClassVar, Final

import httpx

from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings

DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
TIMEOUT_S: Final = 900.0
"""Reasoning models can think for minutes over a dense panel."""


class ProviderError(RuntimeError):
    """A non-2xx answer or a malformed body from the endpoint."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class OpenAICompatProvider:
    """Any chat-completions server that speaks the OpenAI wire protocol.

    ``settings.openai_base_url`` selects the server (unset means api.openai.com) and
    ``settings.openai_api_key`` is sent as the bearer token. Images go as data URLs in
    ``image_url`` content parts. No ``response_format`` is requested, because support for
    JSON-schema outputs differs across the models behind one OpenAI-compatible router; the
    prompt asks for JSON and the caller parses it. When the server is OpenRouter, usage
    accounting is requested so the billed cost lands in ``Usage.cost_usd``.
    """

    name: ClassVar[str] = "openai"

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self._transport = transport

    @property
    def base_url(self) -> str:
        """The chat-completions server, without a trailing slash."""
        return (self.settings.openai_base_url or DEFAULT_BASE_URL).rstrip("/")

    def _payload(self, request: VisionRequest) -> dict[str, Any]:
        """Build the chat-completions body for one request."""
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for image in request.images:
            data = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.media_type};base64,{data}"},
                }
            )
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": content})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if "openrouter.ai" in self.base_url:
            payload["usage"] = {"include": True}
        return {**payload, **request.extra}

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one chat completion against the configured base URL.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The model's answer with token usage, billed cost (OpenRouter) and latency.

        Raises:
            ProviderError: On a non-2xx answer, an error body or a response without choices.
        """
        headers = {"Content-Type": "application/json"}
        if self.settings.openai_api_key is not None:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key.get_secret_value()}"
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=self._payload(request), headers=headers
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ProviderError(response.status_code, response.text[:500])
        body: dict[str, Any] = response.json()
        if "error" in body or not body.get("choices"):
            raise ProviderError(response.status_code, str(body.get("error", body))[:500])
        message = body["choices"][0].get("message") or {}
        usage = body.get("usage") or {}
        cost = usage.get("cost")
        return VisionResponse(
            text=str(message.get("content") or ""),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cost_usd=float(cost) if cost is not None else None,
            ),
            latency_ms=latency_ms,
            raw={
                "id": body.get("id"),
                "model": body.get("model"),
                "provider": body.get("provider"),
                "finish_reason": body["choices"][0].get("finish_reason"),
            },
        )
