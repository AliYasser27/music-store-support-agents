"""Tests for identity resolution in the customer verification node."""

from __future__ import annotations

from sqlalchemy import text

from src.db import engine
from src.nodes.verify_customer import _normalize_phone, _try_resolve_customer_id


def _reference_customer() -> dict:
    """Fetch a real customer row from Chinook to test against."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT CustomerId, Email, Phone FROM Customer WHERE CustomerId = 1")
        ).fetchone()
    assert row is not None, "expected Chinook customer 1 to exist"
    return {"CustomerId": row[0], "Email": row[1], "Phone": row[2]}


def test_resolve_by_explicit_customer_id() -> None:
    ref = _reference_customer()
    assert _try_resolve_customer_id(f"My customer id is {ref['CustomerId']}") == str(
        ref["CustomerId"]
    )


def test_resolve_by_bare_numeric_id() -> None:
    ref = _reference_customer()
    assert _try_resolve_customer_id(str(ref["CustomerId"])) == str(ref["CustomerId"])


def test_resolve_by_email_is_case_insensitive() -> None:
    ref = _reference_customer()
    assert _try_resolve_customer_id(f"my email is {ref['Email'].upper()}") == str(ref["CustomerId"])


def test_resolve_by_phone_with_formatting() -> None:
    ref = _reference_customer()
    assert _try_resolve_customer_id(f"My phone number is {ref['Phone']}.") == str(ref["CustomerId"])


def test_resolve_by_phone_without_country_code() -> None:
    """The last nine digits still match when the country code is omitted."""
    ref = _reference_customer()
    last_nine = _normalize_phone(ref["Phone"])[-9:]
    assert _try_resolve_customer_id(f"call me on {last_nine}") == str(ref["CustomerId"])


def test_normalize_phone_strips_formatting() -> None:
    assert _normalize_phone("+55 (12) 3923-5555") == "551239235555"
    assert _normalize_phone("55.12.3923.5555") == "551239235555"


def test_unknown_customer_id_does_not_resolve() -> None:
    assert _try_resolve_customer_id("my customer id is 999999") is None


def test_unknown_email_does_not_resolve() -> None:
    assert _try_resolve_customer_id("nobody@example.com") is None


def test_garbage_text_does_not_resolve() -> None:
    assert _try_resolve_customer_id("I like the Rolling Stones a lot") is None
