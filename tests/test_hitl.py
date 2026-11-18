"""Tests for the missing-argument human-in-the-loop mechanism.

These exercise ``require_nonblank`` inside a minimal compiled graph, so the
interrupt/resume contract is verified without any LLM calls or API key.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.tools._common import require_nonblank


class _EchoState(TypedDict):
    value: str


def _build_probe_graph():
    """Single-node graph that resolves a blank argument through an interrupt."""

    def resolve(state: _EchoState) -> dict:
        return {"value": require_nonblank(state.get("value"), "Which artist?")}

    builder = StateGraph(_EchoState)
    builder.add_node("resolve", resolve)
    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_interrupt_is_raised_for_blank_argument() -> None:
    graph = _build_probe_graph()
    config = {"configurable": {"thread_id": "hitl-blank"}}

    result = graph.invoke({"value": ""}, config=config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == {"question": "Which artist?"}


def test_resume_supplies_the_missing_argument() -> None:
    graph = _build_probe_graph()
    config = {"configurable": {"thread_id": "hitl-resume"}}

    graph.invoke({"value": "   "}, config=config)
    result = graph.invoke(Command(resume="  The Rolling Stones  "), config=config)

    assert "__interrupt__" not in result
    assert result["value"] == "The Rolling Stones"


def test_nonblank_argument_skips_the_interrupt() -> None:
    graph = _build_probe_graph()
    result = graph.invoke({"value": "Jazz"}, config={"configurable": {"thread_id": "hitl-direct"}})

    assert "__interrupt__" not in result
    assert result["value"] == "Jazz"
