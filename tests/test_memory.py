"""Tests for the per-customer long-term preference store."""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from src.nodes.memory import _NO_PREFERENCES, _PREFERENCES_KEY, _namespace, load_memory_node


def test_load_memory_returns_default_without_customer() -> None:
    store = InMemoryStore()
    assert load_memory_node({"customer_id": "", "messages": []}, {}, store=store) == {
        "loaded_memory": _NO_PREFERENCES
    }


def test_load_memory_returns_default_for_new_customer() -> None:
    store = InMemoryStore()
    assert load_memory_node({"customer_id": "1", "messages": []}, {}, store=store) == {
        "loaded_memory": _NO_PREFERENCES
    }


def test_load_memory_reads_seeded_preferences() -> None:
    store = InMemoryStore()
    store.put(_namespace("1"), _PREFERENCES_KEY, {"text": "Likes The Rolling Stones and Rock."})

    result = load_memory_node({"customer_id": "1", "messages": []}, {}, store=store)

    assert result["loaded_memory"] == "Likes The Rolling Stones and Rock."


def test_preferences_are_isolated_per_customer() -> None:
    store = InMemoryStore()
    store.put(_namespace("1"), _PREFERENCES_KEY, {"text": "Likes Jazz."})

    result = load_memory_node({"customer_id": "2", "messages": []}, {}, store=store)

    assert result["loaded_memory"] == _NO_PREFERENCES
