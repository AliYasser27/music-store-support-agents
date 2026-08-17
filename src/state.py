"""Graph state schema shared across all nodes and sub-agents."""

from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import NotRequired, TypedDict


class State(TypedDict):
    """Short-term (per-thread) memory carried through the graph.

    Attributes:
        customer_id: The verified Chinook CustomerId for this session.
            Empty string until verification succeeds.
        messages: Full conversation history (auto-merged via add_messages).
        loaded_memory: A short natural-language summary of the customer's
            long-term music preferences, injected into agent prompts.
        remaining_steps: Managed value LangGraph uses to guard against
            infinite agent loops.
        route: Internal scratch field holding the supervisor's most recent
            routing decision ("music_catalog" | "invoice_info" | "both" |
            "FINISH"), used only to drive conditional edges. Not part of the
            three required fields, but needed to make routing explicit and
            testable.
    """

    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps
    route: NotRequired[str]
