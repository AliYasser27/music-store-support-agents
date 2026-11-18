"""Human-in-the-loop prompt helpers.

Thin wrappers around ``langgraph.types.interrupt`` so every pause in the
workflow is phrased consistently and carries a machine-readable ``type``.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def ask_for_identity() -> str:
    """Pause the graph and ask the customer to verify their identity.

    Returns:
        The text the customer supplies on resume (a customer id, phone number,
        or email address).
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
    """Pause the graph and ask a free-form clarifying question.

    Args:
        question: The question shown to the customer.

    Returns:
        The text the customer supplies on resume.
    """
    payload: dict[str, Any] = {"type": "clarification", "question": question}
    return str(interrupt(payload)).strip()
