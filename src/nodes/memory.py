"""Long-term memory nodes backed by langgraph.store.memory.InMemoryStore.

Preferences are namespaced per customer: ("customer", customer_id,
"preferences"). We store a single key "summary" holding a concise,
deduplicated natural-language description of the customer's known music
preferences (favorite artists/genres).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.state import State

_PREFERENCES_KEY = "summary"


def _namespace(customer_id: str) -> tuple[str, str, str]:
    return ("customer", customer_id, "preferences")


def load_memory_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Load the customer's saved preferences into short-term state.

    Args:
        state: Current graph state (must already contain a verified
            customer_id).
        config: LangGraph runnable config (unused here, required by the
            store-injection calling convention).
        store: The long-term memory store, auto-injected by LangGraph.

    Returns:
        A partial state update setting ``loaded_memory``.
    """
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return {"loaded_memory": "No preferences on file yet."}

    item = store.get(_namespace(customer_id), _PREFERENCES_KEY)
    if item and item.value.get("text"):
        return {"loaded_memory": item.value["text"]}
    return {"loaded_memory": "No preferences on file yet."}


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

    Args:
        state: Current graph state.
        config: LangGraph runnable config (unused here).
        store: The long-term memory store, auto-injected by LangGraph.

    Returns:
        An empty partial state update (this node only has the side effect
        of writing to the long-term store).
    """
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return {}

    existing_item = store.get(_namespace(customer_id), _PREFERENCES_KEY)
    existing_text = existing_item.value.get("text", "") if existing_item else ""

    recent_messages = state["messages"][-6:]
    conversation_snippet = "\n".join(
        f"{getattr(m, 'type', 'unknown')}: {m.content}"
        for m in recent_messages
        if isinstance(m.content, str)
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(ExtractedPreferences)
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
        extracted = structured_llm.invoke(extraction_prompt)
    except Exception as exc:  # noqa: BLE001 - never let memory extraction crash the turn
        print(f"[save_memory] preference extraction failed, skipping save: {exc}")
        return {}

    if extracted.has_preferences and extracted.summary.strip():
        store.put(
            _namespace(customer_id),
            _PREFERENCES_KEY,
            {"text": extracted.summary.strip()},
        )

    return {}
