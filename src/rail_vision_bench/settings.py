"""Runtime settings read from the environment and an optional ``.env`` file.

Provider credentials are ``SecretStr`` so they never leak into logs or reprs;
the ``RVB_*`` knobs configure the bench itself. ``.env.example`` lists every
key with an empty value and a test keeps that file and ``SETTINGS_ENV_KEYS``
in sync.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration; every field maps to one upper-case env key."""

    # env_ignore_empty: a blank ``KEY=`` line (as in .env.example) means "unset", not "",
    # which would otherwise fail validation of the bool and int fields.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    google_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "rail-vision-bench"
    mlflow_tracking_uri: str | None = None
    hf_token: SecretStr | None = None
    rvb_data_dir: Path = Path("data")
    rvb_runs_dir: Path = Path("runs")
    rvb_harness_host: str = "127.0.0.1"
    rvb_harness_port: int = 8000
    rvb_log_level: str = "INFO"


SETTINGS_ENV_KEYS: Final[tuple[str, ...]] = tuple(key.upper() for key in Settings.model_fields)
"""The environment keys read by :class:`Settings`, in field order (same set as ``.env.example``)."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings.

    The instance is cached; tests call ``get_settings.cache_clear()`` after
    changing the environment.

    Returns:
        The settings built from the environment and ``.env``.
    """
    return Settings()
