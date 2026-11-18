"""Shared helpers for the database-backed agent tools."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    """Convert a SQLAlchemy result set into a list of plain dictionaries."""
    return [dict(row._mapping) for row in rows]


def require_nonblank(value: str | None, question: str) -> str:
    """Return a stripped argument value, asking the customer if it is blank.

    This is the second human-in-the-loop mechanism in the system. When the
    model calls a tool without a required argument (for example, no artist
    name), the tool pauses the graph with ``interrupt()``. The caller resumes
    the graph with the customer's answer, which becomes the argument value.

    Args:
        value: The argument supplied by the calling agent.
        question: The clarifying question shown to the customer.

    Returns:
        The non-blank, stripped argument value.
    """
    if value and str(value).strip():
        return str(value).strip()
    reply = interrupt({"question": question})
    return str(reply).strip()
