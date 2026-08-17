"""Human-in-the-loop helper utilities.

These helpers wrap ``langgraph.types.interrupt`` so that verification and
clarification prompts are phrased consistently across the graph.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def ask_for_identity() -> str:
    """Pause the graph and ask the user to verify their identity.

    Returns:
        The raw string the user supplies on resume (expected to be a
        Customer ID, phone number, or email address).
    """
    payload: dict[str, Any] = {
        "type": "identity_verification",
        "question": (
            "I couldn't verify your identity. Please provide your "
            "Customer ID, Phone, or Email so I can verify your identity."
        ),
    }
    return str(interrupt(payload)).strip()


def ask_for_clarification(question: str) -> str:
    """Pause the graph and ask the user a free-form clarifying question.

    Args:
        question: The clarification question to display to the user.

    Returns:
        The raw string the user supplies on resume.
    """
    payload: dict[str, Any] = {"type": "clarification", "question": question}
    return str(interrupt(payload)).strip()
