"""Graph state schema shared by every node and sub-agent."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import NotRequired, TypedDict


class State(TypedDict):
    """Short-term (per-thread) memory carried through the graph.

    Attributes:
        customer_id: The verified Chinook CustomerId for this session.
            Empty until verification succeeds.
        messages: Full conversation history, merged via ``add_messages``.
        loaded_memory: Summary of the customer's long-term music preferences,
            injected into the sub-agent prompts.
        remaining_steps: Managed value LangGraph uses to guard against
            runaway agent loops.
        route: Internal scratch field holding the supervisor's last decision
            ("music_catalog" | "invoice_info" | "both" | "FINISH"). Used only
            to drive conditional edges.
    """

    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps
    route: NotRequired[str]
