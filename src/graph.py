"""Graph construction for the customer-support workflow.

    START -> verify_customer -> load_memory -> supervisor
                  |                              |-- music_catalog --\
                  |                              |-- invoice_info ----+--> save_memory -> END
                  |                              |-- both: music_catalog -> invoice_info
                  |                              |          -> synthesize -> save_memory -> END
                  |                              |-- FINISH -> END
                  |
                  +-- interrupt() until identity is verified

Nothing here talks to the LLM at import time: the compiled graph is created on
first use by :func:`get_graph`, so importing this module is free and the
offline test suite can inspect the graph without an API key.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from src.agents.invoice_agent import get_invoice_agent
from src.agents.music_agent import get_music_agent
from src.agents.supervisor import build_supervisor_router
from src.llm import get_llm
from src.nodes.memory import load_memory_node, save_memory_node
from src.nodes.verify_customer import verify_customer_node
from src.state import State

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = (
    "Combine the following sub-agent replies into one clear, well-organized "
    "final answer for the customer. Keep all concrete facts (amounts, dates, "
    "album/track names) exactly as given; do not invent anything new."
)


def supervisor_node(state: State) -> dict:
    """Decide which specialist(s) handle the latest message."""
    decision = build_supervisor_router()(state)
    logger.info("route=%s (%s)", decision.route, decision.reasoning)
    return {"route": decision.route}


def music_catalog_node(state: State) -> dict:
    """Run the Music Catalog Sub-Agent on the current state."""
    result = get_music_agent().invoke(state)
    return {"messages": result["messages"]}


def invoice_info_node(state: State) -> dict:
    """Run the Invoice Information Sub-Agent on the current state."""
    result = get_invoice_agent().invoke(state)
    return {"messages": result["messages"]}


def synthesize_node(state: State) -> dict:
    """Merge the two sub-agent replies into one customer-facing answer.

    Only used on the "both" route. Synthesis is best-effort: if the merge call
    fails, the individual sub-agent replies remain in the conversation and are
    still correct, so the turn is not lost.
    """
    recent_ai_texts = [
        message.content
        for message in state["messages"][-6:]
        if isinstance(message, AIMessage)
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    if not recent_ai_texts:
        return {}

    try:
        synthesis = get_llm().invoke(
            [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                *[AIMessage(content=text) for text in recent_ai_texts],
            ]
        )
    except Exception as exc:  # noqa: BLE001 - keep the turn alive if merging fails
        logger.warning("synthesis failed, keeping separate replies: %s", exc)
        return {}

    return {"messages": [synthesis]}


def route_from_supervisor(state: State) -> str:
    """Conditional edge: first destination after the supervisor."""
    route = state.get("route", "FINISH")
    if route in ("music_catalog", "both"):
        return "music_catalog"
    if route == "invoice_info":
        return "invoice_info"
    return "end"


def route_after_music(state: State) -> str:
    """Conditional edge: after the music agent, continue if the route is "both"."""
    return "invoice_info" if state.get("route") == "both" else "save_memory"


def route_after_invoice(state: State) -> str:
    """Conditional edge: after the invoice agent, merge replies on the "both" route."""
    return "synthesize" if state.get("route") == "both" else "save_memory"


def build_graph() -> tuple[StateGraph, InMemoryStore]:
    """Build and compile the workflow.

    Returns:
        ``(graph, store)``. The store is returned so callers can inspect or
        seed long-term memory (handy in tests and demos).
    """
    builder = StateGraph(State)

    builder.add_node("verify_customer", verify_customer_node)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("music_catalog", music_catalog_node)
    builder.add_node("invoice_info", invoice_info_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("save_memory", save_memory_node)

    builder.add_edge(START, "verify_customer")
    builder.add_edge("verify_customer", "load_memory")
    builder.add_edge("load_memory", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"music_catalog": "music_catalog", "invoice_info": "invoice_info", "end": END},
    )
    builder.add_conditional_edges(
        "music_catalog",
        route_after_music,
        {"invoice_info": "invoice_info", "save_memory": "save_memory"},
    )
    builder.add_conditional_edges(
        "invoice_info",
        route_after_invoice,
        {"synthesize": "synthesize", "save_memory": "save_memory"},
    )
    builder.add_edge("synthesize", "save_memory")
    builder.add_edge("save_memory", END)

    store = InMemoryStore()
    compiled = builder.compile(checkpointer=InMemorySaver(), store=store)
    return compiled, store


@lru_cache(maxsize=1)
def get_graph() -> tuple[Any, InMemoryStore]:
    """Return the process-wide compiled graph and its long-term store.

    A single instance is reused so that the checkpointer (short-term memory)
    and store (long-term memory) survive across turns of a conversation, both
    in the CLI and in notebooks.
    """
    return build_graph()
