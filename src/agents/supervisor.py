"""Supervisor agent: routes each customer message to the right specialist(s)."""

from __future__ import annotations

from functools import lru_cache
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
    reasoning: str = Field(description="One short sentence explaining the routing choice.")


@lru_cache(maxsize=1)
def build_supervisor_router():
    """Return a cached function mapping the current state to a RouteDecision.

    The structured-output chain is built once per process rather than once per
    turn, which avoids re-creating the client on every message.
    """
    structured_llm = get_llm().with_structured_output(RouteDecision)

    def route(state: State) -> RouteDecision:
        loaded_memory = state.get("loaded_memory") or "No preferences on file yet."
        system_text = SUPERVISOR_SYSTEM_PROMPT.format(loaded_memory=loaded_memory)
        messages = [SystemMessage(content=system_text), *state["messages"]]
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - a routing failure must not crash the turn
            print(f"[supervisor] routing call failed, defaulting to FINISH: {exc}")
            return RouteDecision(
                route="FINISH", reasoning="Routing call failed; ending turn safely."
            )

    return route
