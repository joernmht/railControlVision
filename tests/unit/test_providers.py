"""Provider registry and request/response models."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from rail_vision_bench.config import PROVIDER_NAMES
from rail_vision_bench.providers.base import ImageInput, Usage, VisionRequest, VisionResponse
from rail_vision_bench.providers.registry import PROVIDERS, get_provider
from rail_vision_bench.settings import Settings


def test_registry_keys_match_provider_names():
    assert set(PROVIDERS) == set(PROVIDER_NAMES)


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_get_provider_resolves_every_name(name: str):
    settings = Settings(_env_file=None)
    provider = get_provider(name, settings)
    assert provider.name == name
    # The provider stores the settings it was built with (credentials come from there).
    assert getattr(provider, "settings", None) is settings


SDK_MODULES = ("anthropic", "openai", "google.genai", "mistralai", "ollama", "litellm")


def test_registry_import_loads_no_sdk():
    """The registry's factories import provider modules (and their SDKs) only when called."""
    code = (
        "import sys, rail_vision_bench.providers, rail_vision_bench.providers.registry\n"
        f"loaded = [m for m in {SDK_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_provider_builds_without_credentials(name: str):
    """Clients are created lazily, so a provider without keys can still be constructed."""
    provider = get_provider(name, Settings(_env_file=None))
    assert getattr(provider, "_client", None) is None
    assert callable(provider.complete)


def test_get_provider_defaults_to_get_settings():
    provider = get_provider("ollama")
    assert provider.name == "ollama"


def test_unknown_provider_lists_known_names():
    with pytest.raises(KeyError, match="anthropic") as exc:
        get_provider("nope")
    assert "unknown provider 'nope'" in str(exc.value)


def test_request_response_round_trip():
    request = VisionRequest(
        model="m",
        prompt="p",
        images=[ImageInput(data=b"\x89PNG", media_type="image/jpeg")],
        system="s",
        response_schema={"type": "object"},
        max_tokens=16,
        temperature=0.5,
    )
    assert VisionRequest.model_validate(request.model_dump()) == request
    response = VisionResponse(
        text="{}", parsed={}, usage=Usage(input_tokens=1, output_tokens=2), latency_ms=3.5
    )
    assert VisionResponse.model_validate(response.model_dump()) == response


def test_defaults():
    request = VisionRequest(model="m", prompt="p")
    assert request.images == []
    assert request.max_tokens == 4096
    assert request.temperature == 0.0
    response = VisionResponse(text="")
    assert response.usage == Usage()
    assert response.parsed is None
    assert response.raw is None


def test_image_input_rejects_gif():
    with pytest.raises(ValidationError):
        ImageInput.model_validate({"data": b"", "media_type": "image/gif"})


def test_models_forbid_extra_keys():
    with pytest.raises(ValidationError):
        VisionRequest.model_validate({"model": "m", "prompt": "p", "extra": 1})
