"""Supervisor agent: routes customer queries to the correct sub-agent(s)."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.state import State

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor for a digital music \
store's customer support system. You do not answer questions yourself -- \
you decide which specialist(s) should handle the customer's latest \
message.

Specialists available:
- "music_catalog": albums, tracks, genres, song lookups, and \
recommendations.
- "invoice_info": invoices, purchase history, prices paid, and which \
support employee handled an invoice.
- "both": the message clearly needs BOTH specialists (e.g. it asks about \
a purchase AND about albums/tracks in the same message).
- "FINISH": the conversation's latest turn has already been fully \
answered by the specialists and nothing further is needed right now.

Customer's known music preferences (for context only, does not affect \
routing): {loaded_memory}

Read the latest customer message and choose exactly one route.
"""


class RouteDecision(BaseModel):
    """Structured routing decision produced by the supervisor."""

    route: Literal["music_catalog", "invoice_info", "both", "FINISH"] = Field(
        description="Which specialist(s) should handle the latest message."
    )
    reasoning: str = Field(
        description="One short sentence explaining the routing choice."
    )


def build_supervisor_router():
    """Build a callable that returns a RouteDecision for the current state."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    def route(state: State) -> RouteDecision:
        loaded_memory = state.get("loaded_memory") or "No preferences on file yet."
        system_text = SUPERVISOR_SYSTEM_PROMPT.format(loaded_memory=loaded_memory)
        messages = [SystemMessage(content=system_text)] + state["messages"]
        try:
            decision = structured_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - surface a safe fallback
            print(f"[supervisor] routing call failed, defaulting to FINISH: {exc}")
            decision = RouteDecision(
                route="FINISH", reasoning="Routing call failed; ending turn safely."
            )
        return decision

    return route
