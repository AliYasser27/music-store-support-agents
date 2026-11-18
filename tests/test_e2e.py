"""End-to-end tests for the two required scenarios.

These drive the full compiled graph, including the LLM-backed sub-agents, so
they need a valid ``GROQ_API_KEY``. They are marked ``e2e`` and excluded from
CI with ``pytest -m "not e2e"``.
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.config import get_settings

requires_llm = pytest.mark.skipif(
    not get_settings().has_llm_credentials,
    reason="GROQ_API_KEY not set; skipping live LLM end-to-end tests.",
)

pytestmark = [pytest.mark.e2e, requires_llm]


def _ai_text(result: dict) -> str:
    """Concatenate every AI message in the thread."""
    return "\n".join(
        message.content
        for message in result.get("messages", [])
        if isinstance(message, AIMessage) and isinstance(message.content, str)
    )


def test_case_1_verified_customer_gets_both_answers(graph_and_store) -> None:
    graph, _ = graph_and_store
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "My phone number is +55 (12) 3923-5555. How much was "
                        "my most recent purchase? What albums do you have by "
                        "the Rolling Stones?"
                    )
                )
            ]
        },
        config=config,
    )

    assert "__interrupt__" not in result
    assert result["customer_id"] == "1"
    assert "rolling stones" in _ai_text(result).lower()


def test_case_1_follow_up_uses_long_term_memory(graph_and_store) -> None:
    graph, _ = graph_and_store
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "My phone number is +55 (12) 3923-5555. What albums do "
                        "you have by the Rolling Stones?"
                    )
                )
            ]
        },
        config=config,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="List some songs that match my preferences?")]},
        config=config,
    )

    assert "__interrupt__" not in result
    assert _ai_text(result).strip()


def test_case_2_unverified_customer_is_interrupted_then_resumes(graph_and_store) -> None:
    graph, _ = graph_and_store
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "How much was my most recent purchase? What albums do "
                        "you have by the Rolling Stones?"
                    )
                )
            ]
        },
        config=config,
    )

    assert "__interrupt__" in result
    question = result["__interrupt__"][0].value["question"].lower()
    assert "customer id" in question or "phone" in question or "email" in question

    resumed = graph.invoke(Command(resume="My customer id is 1"), config=config)

    assert "__interrupt__" not in resumed
    assert resumed["customer_id"] == "1"
    assert _ai_text(resumed).strip()
