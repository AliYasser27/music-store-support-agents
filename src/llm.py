"""LLM construction.

A single factory is used everywhere so that the provider can be swapped in one
place. The client is built lazily (on first use) and cached, which means
importing any module -- including the graph -- never requires an API key. That
is what allows the offline test suite (and CI) to run without credentials.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from src.config import ConfigurationError, get_settings


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, model_name: str | None = None) -> ChatGroq:
    """Return a cached ``ChatGroq`` client.

    Args:
        temperature: Sampling temperature. The default of 0 keeps
            support answers deterministic and repeatable.
        model_name: Optional override of the configured model name.

    Returns:
        A configured ``ChatGroq`` client.

    Raises:
        ConfigurationError: If ``GROQ_API_KEY`` is not set.
    """
    settings = get_settings()
    if not settings.has_llm_credentials:
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export GROQ_API_KEY in your shell."
        )
    return ChatGroq(
        model=model_name or settings.model_name,
        temperature=temperature,
    )
