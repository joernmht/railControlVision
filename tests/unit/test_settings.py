"""Settings: defaults, environment overrides, secret masking, .env.example sync, caching."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rail_vision_bench.settings import SETTINGS_ENV_KEYS, Settings, get_settings
from tests.conftest import REPO_ROOT

ENV_EXAMPLE = REPO_ROOT / ".env.example"


def test_defaults_without_env_file():
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.openai_base_url is None
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.langsmith_tracing is False
    assert settings.langsmith_project == "rail-vision-bench"
    assert settings.rvb_data_dir == Path("data")
    assert settings.rvb_runs_dir == Path("runs")
    assert settings.rvb_harness_host == "127.0.0.1"
    assert settings.rvb_harness_port == 8000
    assert settings.rvb_log_level == "INFO"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RVB_HARNESS_PORT", "9000")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("RVB_DATA_DIR", "/somewhere/data")
    monkeypatch.setenv("rvb_log_level", "DEBUG")  # case_sensitive=False
    settings = Settings(_env_file=None)
    assert settings.rvb_harness_port == 9000
    assert settings.langsmith_tracing is True
    assert settings.rvb_data_dir == Path("/somewhere/data")
    assert settings.rvb_log_level == "DEBUG"


def test_empty_env_value_means_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RVB_HARNESS_PORT", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "")
    assert Settings(_env_file=None).rvb_harness_port == 8000


def test_unknown_env_keys_are_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RVB_NOT_A_SETTING", "1")
    assert not hasattr(Settings(_env_file=None), "rvb_not_a_setting")


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-very-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-very-secret")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is not None
    assert settings.hf_token is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-very-secret"
    for rendered in (repr(settings), str(settings), str(settings.model_dump())):
        assert "sk-very-secret" not in rendered
        assert "hf-very-secret" not in rendered
        assert "**********" in rendered


def test_env_example_matches_settings_keys():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    keys_in_file = re.findall(r"^([A-Z0-9_]+)=$", text, flags=re.MULTILINE)
    assert len(keys_in_file) == len(set(keys_in_file)), "duplicate key in .env.example"
    assert set(keys_in_file) == set(SETTINGS_ENV_KEYS)
    assert len(SETTINGS_ENV_KEYS) == 16


def test_env_example_loads_to_defaults():
    # Every value in the example is empty, so loading it must equal the defaults.
    assert Settings(_env_file=ENV_EXAMPLE) == Settings(_env_file=None)


def test_settings_env_keys_are_upper_and_match_fields():
    assert tuple(name.upper() for name in Settings.model_fields) == SETTINGS_ENV_KEYS
    assert all(key == key.upper() for key in SETTINGS_ENV_KEYS)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch):
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("RVB_LOG_LEVEL", "DEBUG")
    assert get_settings().rvb_log_level == first.rvb_log_level  # still the cached instance
    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed is not first
    assert refreshed.rvb_log_level == "DEBUG"
