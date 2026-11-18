"""Application configuration, read from environment variables.

All environment access goes through this module so that the rest of the code
base never touches ``os.environ`` directly. Values are read lazily and cached,
which keeps imports free of side effects and lets the offline test suite run
without any credentials configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)
DEFAULT_CHINOOK_HTTP_TIMEOUT = 30.0


class ConfigurationError(RuntimeError):
    """Raised when a required piece of configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the process configuration."""

    groq_api_key: str
    model_name: str
    chinook_sql_url: str
    chinook_http_timeout: float

    @property
    def has_llm_credentials(self) -> bool:
        """Whether an LLM API key is available for this process."""
        return bool(self.groq_api_key)


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings, reading the environment once."""
    return Settings(
        groq_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME).strip(),
        chinook_sql_url=os.environ.get("CHINOOK_SQL_URL", DEFAULT_CHINOOK_SQL_URL).strip(),
        chinook_http_timeout=_env_float("CHINOOK_HTTP_TIMEOUT", DEFAULT_CHINOOK_HTTP_TIMEOUT),
    )
