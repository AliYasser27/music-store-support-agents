"""Tests for graph wiring.

These run without an API key: building the graph only registers nodes and
edges, and the LLM clients are created lazily on first invocation. This is the
test that would have caught an import-time API-key requirement.
"""

from __future__ import annotations

from src.graph import build_graph, route_after_invoice, route_after_music, route_from_supervisor

EXPECTED_NODES = {
    "verify_customer",
    "load_memory",
    "supervisor",
    "music_catalog",
    "invoice_info",
    "synthesize",
    "save_memory",
}


def test_graph_builds_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    graph, store = build_graph()
    assert graph is not None
    assert store is not None


def test_graph_contains_all_nodes(graph_and_store) -> None:
    graph, _ = graph_and_store
    assert set(graph.get_graph().nodes) == EXPECTED_NODES | {"__start__", "__end__"}


def test_supervisor_routes_to_music_first() -> None:
    assert route_from_supervisor({"route": "music_catalog"}) == "music_catalog"
    assert route_from_supervisor({"route": "both"}) == "music_catalog"
    assert route_from_supervisor({"route": "invoice_info"}) == "invoice_info"
    assert route_from_supervisor({"route": "FINISH"}) == "end"


def test_both_route_runs_invoice_then_synthesis() -> None:
    assert route_after_music({"route": "both"}) == "invoice_info"
    assert route_after_invoice({"route": "both"}) == "synthesize"


def test_single_routes_go_straight_to_memory() -> None:
    assert route_after_music({"route": "music_catalog"}) == "save_memory"
    assert route_after_invoice({"route": "invoice_info"}) == "save_memory"
