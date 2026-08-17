"""Shared LLM configuration.

Centralizing model construction here makes it easy to swap providers
without touching every agent module. Currently configured for **Groq**.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")


def get_llm(temperature: float = 0.0, model_name: str | None = None) -> ChatGroq:
    """Construct a ChatGroq instance.

    Args:
        temperature: Sampling temperature. Defaults to 0 for deterministic,
            support-oriented responses.
        model_name: Optional override of the default model name. Falls back
            to the MODEL_NAME env var, then "llama-3.3-70b-versatile".

    Returns:
        A configured ChatGroq instance. Reads the API key from the
        GROQ_API_KEY environment variable.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and set "
            "your key, or export it in your shell before running."
        )
    return ChatGroq(model=model_name or DEFAULT_MODEL_NAME, temperature=temperature)
