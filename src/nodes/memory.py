"""Long-term memory nodes, backed by ``langgraph.store.memory.InMemoryStore``.

Preferences are namespaced per customer under
``("customer", customer_id, "preferences")`` with a single ``"summary"`` key
holding a concise description of the customer's favourite artists and genres.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.state import State

logger = logging.getLogger(__name__)

_PREFERENCES_KEY = "summary"
_NO_PREFERENCES = "No preferences on file yet."


def _namespace(customer_id: str) -> tuple[str, str, str]:
    """Return the long-term-memory namespace for a customer."""
    return ("customer", customer_id, "preferences")


def load_memory_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Load the customer's saved preferences into short-term state.

    Args:
        state: Current graph state; ``customer_id`` must already be verified.
        config: LangGraph runnable config (required by the node signature).
        store: Long-term store, injected by LangGraph.

    Returns:
        A partial state update setting ``loaded_memory``.
    """
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return {"loaded_memory": _NO_PREFERENCES}

    item = store.get(_namespace(customer_id), _PREFERENCES_KEY)
    if item and item.value.get("text"):
        return {"loaded_memory": item.value["text"]}
    return {"loaded_memory": _NO_PREFERENCES}


class ExtractedPreferences(BaseModel):
    """Structured extraction of music preferences from a conversation."""

    has_preferences: bool = Field(
        description="Whether the conversation reveals any concrete music "
        "preference (specific artist, genre, or song the customer likes)."
    )
    summary: str = Field(
        default="",
        description="A short (<=25 words), deduplicated, comma-separated "
        "summary of the customer's favorite artists/genres, merging any "
        "prior known preferences with new ones. Empty if has_preferences "
        "is False.",
    )


def save_memory_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract and persist music preferences revealed in the conversation.

    The extraction is best-effort: a failure here is logged and skipped so a
    memory problem can never break a customer-facing turn.

    Args:
        state: Current graph state.
        config: LangGraph runnable config (required by the node signature).
        store: Long-term store, injected by LangGraph.

    Returns:
        An empty partial state update; this node only has a side effect.
    """
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return {}

    existing_item = store.get(_namespace(customer_id), _PREFERENCES_KEY)
    existing_text = existing_item.value.get("text", "") if existing_item else ""

    conversation_snippet = "\n".join(
        f"{getattr(message, 'type', 'unknown')}: {message.content}"
        for message in state["messages"][-6:]
        if isinstance(message.content, str)
    )

    extraction_prompt = [
        SystemMessage(
            content=(
                "Extract the customer's music preferences (artists, genres, "
                "songs they like) from the conversation below. Merge with "
                "any previously known preferences, deduplicate, and keep the "
                "summary concise (<=25 words).\n\n"
                f"Previously known preferences: {existing_text or 'none'}"
            )
        ),
        HumanMessage(content=conversation_snippet or "(no content)"),
    ]

    try:
        extracted = get_llm().with_structured_output(ExtractedPreferences).invoke(extraction_prompt)
    except Exception as exc:  # noqa: BLE001 - never let memory extraction crash a turn
        logger.warning("preference extraction failed, skipping save: %s", exc)
        return {}

    if extracted.has_preferences and extracted.summary.strip():
        store.put(
            _namespace(customer_id),
            _PREFERENCES_KEY,
            {"text": extracted.summary.strip()},
        )
        logger.info("saved preferences for customer %s", customer_id)

    return {}
