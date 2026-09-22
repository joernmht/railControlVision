"""Provider registry: name -> lazy factory, so importing the registry loads no SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from rail_vision_bench.providers.base import VisionProvider
from rail_vision_bench.settings import Settings, get_settings

# Each factory imports its provider module inside the function body: the six
# modules stay unimported until a run actually selects them.


def _anthropic(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.claude import AnthropicProvider

    return AnthropicProvider(settings)


def _openai(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(settings)


def _gemini(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.gemini import GeminiProvider

    return GeminiProvider(settings)


def _mistral(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.mistral import MistralProvider

    return MistralProvider(settings)


def _ollama(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.ollama_local import OllamaProvider

    return OllamaProvider(settings)


def _litellm(settings: Settings) -> VisionProvider:
    from rail_vision_bench.providers.litellm_router import LiteLLMProvider

    return LiteLLMProvider(settings)


PROVIDERS: Final[dict[str, Callable[[Settings], VisionProvider]]] = {
    "anthropic": _anthropic,
    "openai": _openai,
    "gemini": _gemini,
    "mistral": _mistral,
    "ollama": _ollama,
    "litellm": _litellm,
}
"""Factories keyed by provider name; the key set equals ``config.PROVIDER_NAMES``."""


def get_provider(name: str, settings: Settings | None = None) -> VisionProvider:
    """Instantiate the provider registered under ``name``.

    Args:
        name: A key of :data:`PROVIDERS` (the ``provider`` field of a catalogue entry).
        settings: Credentials and hosts; ``get_settings()`` when omitted.

    Returns:
        A provider bound to the settings.

    Raises:
        KeyError: If ``name`` is unknown; the message lists the known names.
    """
    try:
        factory = PROVIDERS[name]
    except KeyError:
        msg = f"unknown provider {name!r}; known: {sorted(PROVIDERS)}"
        raise KeyError(msg) from None
    return factory(settings if settings is not None else get_settings())
