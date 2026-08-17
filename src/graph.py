"""Builds and compiles the full customer-support StateGraph.

Graph shape (see README.md for the ASCII diagram):

    START -> verify_customer -> load_memory -> supervisor
                                                    |-- music_catalog --\
                                                    |-- invoice_info ----+--> save_memory -> END
                                                    |-- both: music_catalog -> invoice_info -> synthesize -> save_memory -> END
                                                    |-- FINISH -> END
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from src.agents.invoice_agent import invoice_agent
from src.agents.music_agent import music_agent
from src.agents.supervisor import build_supervisor_router
from src.llm import get_llm
from src.nodes.memory import load_memory_node, save_memory_node
from src.nodes.verify_customer import verify_customer_node
from src.state import State

_route_query = build_supervisor_router()


def supervisor_node(state: State) -> dict:
    """Decide routing for the latest customer message and record it in state."""
    decision = _route_query(state)
    print(f"[supervisor] route={decision.route} ({decision.reasoning})")
    return {"route": decision.route}


def music_catalog_node(state: State) -> dict:
    """Run the Music Catalog Sub-Agent on the current state."""
    result = music_agent.invoke(state)
    return {"messages": result["messages"]}


def invoice_info_node(state: State) -> dict:
    """Run the Invoice Information Sub-Agent on the current state."""
    result = invoice_agent.invoke(state)
    return {"messages": result["messages"]}


def synthesize_node(state: State) -> dict:
    """Combine the music + invoice sub-agent replies into one final answer.

    Only used when the supervisor routed to "both". Looks back over the
    most recent AI messages produced by the two sub-agents and asks the LLM
    to merge them into a single, well-organized customer-facing reply.
    """
    recent_ai_texts = [
        m.content
        for m in state["messages"][-6:]
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
    ]
    if not recent_ai_texts:
        return {}

    llm = get_llm()
    try:
        synthesis = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Combine the following sub-agent replies into one "
                        "clear, well-organized final answer for the "
                        "customer. Keep all concrete facts (amounts, dates, "
                        "album/track names) exactly as given; do not "
                        "invent anything new."
                    )
                ),
                *[AIMessage(content=t) for t in recent_ai_texts],
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[synthesize] LLM call failed, skipping synthesis: {exc}")
        return {}

    return {"messages": [synthesis]}


def route_from_supervisor(state: State) -> str:
    """Conditional edge: choose the first destination after the supervisor."""
    route = state.get("route", "FINISH")
    if route in ("music_catalog", "both"):
        return "music_catalog"
    if route == "invoice_info":
        return "invoice_info"
    return "end"


def route_after_music(state: State) -> str:
    """Conditional edge: after music_catalog, continue to invoice_info if 'both'."""
    if state.get("route") == "both":
        return "invoice_info"
    return "save_memory"


def route_after_invoice(state: State) -> str:
    """Conditional edge: after invoice_info, synthesize if this was a 'both' route."""
    if state.get("route") == "both":
        return "synthesize"
    return "save_memory"


def build_graph():
    """Construct and compile the full multi-agent StateGraph.

    Returns:
        A tuple of (compiled_graph, store) so callers can inspect/seed the
        long-term memory store directly if needed (e.g. in tests).
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

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph, store


graph, memory_store = build_graph()
