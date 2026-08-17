"""Tests for the customer verification node's identity-resolution logic."""

from __future__ import annotations

from sqlalchemy import text

from src.db import engine
from src.nodes.verify_customer import _try_resolve_customer_id


def _get_reference_customer() -> dict:
    """Fetch a real customer row to test against (CustomerId=1)."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT CustomerId, Email, Phone FROM Customer WHERE CustomerId = 1")
        ).fetchone()
    assert row is not None, "expected Chinook Customer with CustomerId=1 to exist"
    return {"CustomerId": row[0], "Email": row[1], "Phone": row[2]}


def test_resolve_by_explicit_customer_id() -> None:
    ref = _get_reference_customer()
    resolved = _try_resolve_customer_id(f"My customer id is {ref['CustomerId']}")
    assert resolved == str(ref["CustomerId"])


def test_resolve_by_bare_numeric_id() -> None:
    ref = _get_reference_customer()
    resolved = _try_resolve_customer_id(str(ref["CustomerId"]))
    assert resolved == str(ref["CustomerId"])


def test_resolve_by_email() -> None:
    ref = _get_reference_customer()
    resolved = _try_resolve_customer_id(f"my email is {ref['Email']}")
    assert resolved == str(ref["CustomerId"])


def test_resolve_by_phone_with_formatting() -> None:
    ref = _get_reference_customer()
    # Reformat the real phone number with extra punctuation/spacing to
    # confirm normalization works.
    messy_phone = ref["Phone"]
    resolved = _try_resolve_customer_id(f"My phone number is {messy_phone}.")
    assert resolved == str(ref["CustomerId"])


def test_resolve_returns_none_for_garbage() -> None:
    resolved = _try_resolve_customer_id("I like the Rolling Stones a lot")
    assert resolved is None
