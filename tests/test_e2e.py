"""End-to-end tests for the two required test cases (section 12 of the spec).

These tests drive the full compiled graph, including LLM-backed sub-agents
and the supervisor, so they require a valid OPENAI_API_KEY. They are
skipped automatically if one is not configured.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping LLM-backed end-to-end tests.",
)


def _ai_text(result: dict) -> str:
    texts = [
        m.content
        for m in result.get("messages", [])
        if isinstance(m, AIMessage) and isinstance(m.content, str)
    ]
    return "\n".join(texts)


@requires_openai_key
def test_case_1_verified_customer_both_routes() -> None:
    from src.graph import graph

    thread_id = f"test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

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

    # Should not need to interrupt for identity since the phone matches a
    # known Chinook customer.
    assert "__interrupt__" not in result
    assert result["customer_id"], "expected verification to resolve a customer_id"

    combined_text = _ai_text(result).lower()
    assert "rolling stones" in combined_text or "voodoo lounge" in combined_text

    # Follow-up turn: preference-based recommendation using long-term memory.
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="List some songs that match my preferences?")]},
        config=config,
    )
    assert "__interrupt__" not in result2
    follow_up_text = _ai_text(result2).lower()
    assert len(follow_up_text) > 0


@requires_openai_key
def test_case_2_unverified_customer_triggers_interrupt() -> None:
    from src.graph import graph

    thread_id = f"test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "How much was my most recent purchase? What albums "
                        "do you have by the Rolling Stones?"
                    )
                )
            ]
        },
        config=config,
    )

    assert "__interrupt__" in result
    question = result["__interrupt__"][0].value.get("question", "")
    assert "customer id" in question.lower() or "phone" in question.lower() or "email" in question.lower()
