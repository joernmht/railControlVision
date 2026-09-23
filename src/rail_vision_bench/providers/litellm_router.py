"""LiteLLM unified router over many hosted and local models."""

from __future__ import annotations

from typing import Any, ClassVar, Final, NamedTuple

import litellm
from litellm import exceptions as litellm_errors

from rail_vision_bench.providers._common import (
    call_with_retry,
    data_url,
    dump_raw,
    is_transient_status,
    is_transport_error,
    parse_answer,
    require_secret,
    schema_name,
)
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings

_TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    litellm_errors.RateLimitError,
    litellm_errors.InternalServerError,
    litellm_errors.ServiceUnavailableError,
    litellm_errors.APIConnectionError,
    litellm_errors.Timeout,
)


class _Route(NamedTuple):
    """Which settings field authenticates a LiteLLM provider prefix."""

    key_field: str | None
    env_key: str | None
    base_field: str | None


ROUTES: Final[dict[str, _Route]] = {
    "anthropic": _Route("anthropic_api_key", "ANTHROPIC_API_KEY", None),
    "openai": _Route("openai_api_key", "OPENAI_API_KEY", "openai_base_url"),
    "gemini": _Route("google_api_key", "GOOGLE_API_KEY", None),
    "mistral": _Route("mistral_api_key", "MISTRAL_API_KEY", None),
    "ollama": _Route(None, None, "ollama_host"),
    "ollama_chat": _Route(None, None, "ollama_host"),
}
"""LiteLLM ``provider/`` prefixes whose credentials come from :class:`Settings`."""


def _is_transient(exc: BaseException) -> bool:
    """Retry LiteLLM's rate-limit, 5xx, timeout and connection exceptions."""
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    if isinstance(exc, litellm_errors.APIError):
        return is_transient_status(getattr(exc, "status_code", None))
    return is_transport_error(exc)


class LiteLLMProvider:
    """One interface for every backend LiteLLM knows.

    Calls ``litellm.acompletion`` with the model id in LiteLLM's
    ``provider/model`` notation, the images as data-URL ``image_url`` content
    parts before the prompt text and ``request.system`` as a system message.
    A ``response_schema`` becomes a non-strict ``json_schema``
    ``response_format`` (LiteLLM translates it per backend, e.g. into a tool
    call for Anthropic); ``drop_params`` lets LiteLLM drop parameters a backend
    does not accept. For the prefixes in :data:`ROUTES` the key (and base URL)
    comes from the same settings the dedicated providers use and a missing key
    fails fast; other prefixes fall back to LiteLLM's own environment lookup.
    ``Usage.cost_usd`` stays None like every other provider: the runner prices
    tokens from the model catalogue.
    """

    name: ClassVar[str] = "litellm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _credentials(self, model: str) -> dict[str, str]:
        """Resolve ``api_key``/``api_base`` for a ``provider/model`` id from the settings."""
        prefix, sep, _ = model.partition("/")
        route = ROUTES.get(prefix) if sep else None
        if route is None:
            return {}
        credentials: dict[str, str] = {}
        if route.key_field is not None and route.env_key is not None:
            credentials["api_key"] = require_secret(
                getattr(self.settings, route.key_field), route.env_key, f"{self.name} ({prefix})"
            )
        if route.base_field is not None:
            base = getattr(self.settings, route.base_field)
            if base:
                credentials["api_base"] = str(base)
        return credentials

    async def complete(self, request: VisionRequest) -> VisionResponse:
        """Run one completion through LiteLLM.

        Args:
            request: The prompt, images and decoding parameters.

        Returns:
            The first choice's text, the JSON recovered from it, token usage and
            the latency of the successful attempt.

        Raises:
            ValueError: If the model's provider prefix needs a key that is not configured.
            litellm.exceptions.APIError: If the call fails permanently or retries run out
                (LiteLLM maps every backend error onto its OpenAI-style exceptions).
        """
        credentials = self._credentials(request.model)
        parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in request.images
        ]
        parts.append({"type": "text", "text": request.prompt})
        messages: list[dict[str, Any]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": parts})
        extra: dict[str, Any] = {}
        if request.response_schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name(request.response_schema),
                    "schema": request.response_schema,
                    "strict": False,
                },
            }

        async def call() -> Any:
            return await litellm.acompletion(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                drop_params=True,
                **credentials,
                **extra,
            )

        response, latency_ms = await call_with_retry(call, _is_transient)
        choices = getattr(response, "choices", None) or []
        text = (getattr(choices[0].message, "content", None) or "") if choices else ""
        usage = getattr(response, "usage", None)
        return VisionResponse(
            text=text,
            parsed=parse_answer(text),
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            raw=dump_raw(response),
        )
