"""Provider registry: name -> lazy factory, so importing the registry loads no SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

from rail_vision_bench.providers.base import VisionProvider
from rail_vision_bench.settings import Settings, get_settings

if TYPE_CHECKING:
    from rail_vision_bench.config import ModelConfig

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


SETTINGS_PARAMS: Final[dict[str, dict[str, str]]] = {
    "openai": {"base_url": "openai_base_url"},
    "litellm": {"base_url": "openai_base_url"},
    "ollama": {"host": "ollama_host"},
}
"""Catalogue ``params`` keys that override a settings field for one model, per provider."""


def settings_for_model(model: ModelConfig, settings: Settings) -> Settings:
    """Apply a catalogue entry's endpoint ``params`` on top of the settings.

    ``params.base_url`` of an ``openai`` (or ``litellm``) entry replaces
    ``OPENAI_BASE_URL`` and ``params.host`` of an ``ollama`` entry replaces
    ``OLLAMA_HOST``, so one catalogue can mix a hosted model with a local
    OpenAI-compatible server. A null value keeps the environment's setting;
    other params are left to the provider.

    Args:
        model: The catalogue entry.
        settings: The process settings.

    Returns:
        The settings for this model (``settings`` itself when nothing overrides).
    """
    update: dict[str, Any] = {
        field: model.params[key]
        for key, field in SETTINGS_PARAMS.get(model.provider, {}).items()
        if model.params.get(key) is not None
    }
    return settings.model_copy(update=update) if update else settings


def provider_for_model(model: ModelConfig, settings: Settings) -> VisionProvider:
    """Instantiate the provider of a catalogue entry, honouring its endpoint params.

    Args:
        model: The catalogue entry.
        settings: The process settings.

    Returns:
        The provider bound to :func:`settings_for_model`.
    """
    return get_provider(model.provider, settings_for_model(model, settings))
