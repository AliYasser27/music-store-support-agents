"""Shared pytest fixtures.

The Chinook database is loaded once at import time by ``src.db`` and reused by
every test, so these fixtures only handle graph construction.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def graph_and_store():
    """Return the compiled graph and its long-term store."""
    from src.graph import get_graph

    return get_graph()
