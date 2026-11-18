"""Customer verification node.

The node resolves a Chinook ``CustomerId`` from the latest message, accepting
an explicit customer id, an email address, or a phone number. If none of those
resolve, the graph interrupts and asks for credentials, repeating until the
customer is verified. The resolved id then lives in short-term state for the
rest of the thread.
"""

from __future__ import annotations

import re

from sqlalchemy import text

from src.db import engine
from src.nodes.hitl import ask_for_identity
from src.state import State

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CUSTOMER_ID_RE = re.compile(r"\bcustomer\s*id\D{0,5}(\d+)\b", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^\s*(\d{1,10})\s*$")
_PHONE_RE = re.compile(r"[+()\d][\d()+\-.\s]{5,}\d")


def _normalize_phone(raw: str) -> str:
    """Strip separators and a leading '+' from a phone number."""
    digits = re.sub(r"[\s()\-.]", "", raw)
    return digits.lstrip("+")


def _lookup_by_customer_id(customer_id: str) -> str | None:
    query = text("SELECT CustomerId FROM Customer WHERE CustomerId = :cid")
    with engine.connect() as conn:
        row = conn.execute(query, {"cid": customer_id}).fetchone()
    return str(row[0]) if row else None


def _lookup_by_email(email: str) -> str | None:
    query = text("SELECT CustomerId FROM Customer WHERE LOWER(Email) = LOWER(:email)")
    with engine.connect() as conn:
        row = conn.execute(query, {"email": email}).fetchone()
    return str(row[0]) if row else None


def _lookup_by_phone(phone: str) -> str | None:
    """Match a normalized phone number against the Chinook customer table.

    Numbers are compared on their normalized digits first, then on the last
    nine digits to tolerate differing country-code prefixes.
    """
    normalized_input = _normalize_phone(phone)
    if not normalized_input:
        return None

    query = text("SELECT CustomerId, Phone FROM Customer WHERE Phone IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    for customer_id, db_phone in rows:
        normalized_db_phone = _normalize_phone(db_phone)
        if normalized_db_phone == normalized_input:
            return str(customer_id)
        if len(normalized_db_phone) >= 9 and normalized_input.endswith(normalized_db_phone[-9:]):
            return str(customer_id)
    return None


def _try_resolve_customer_id(user_text: str) -> str | None:
    """Try to resolve a Chinook CustomerId from free-form user text.

    Checks, in order: an explicit "customer id" mention, a bare numeric
    reply, an email address, then a phone number.
    """
    if not user_text:
        return None

    match = _CUSTOMER_ID_RE.search(user_text)
    if match and (resolved := _lookup_by_customer_id(match.group(1))):
        return resolved

    match = _BARE_ID_RE.match(user_text)
    if match and (resolved := _lookup_by_customer_id(match.group(1))):
        return resolved

    match = _EMAIL_RE.search(user_text)
    if match and (resolved := _lookup_by_email(match.group(0))):
        return resolved

    match = _PHONE_RE.search(user_text)
    if match and (resolved := _lookup_by_phone(match.group(0))):
        return resolved

    return None


def verify_customer_node(state: State) -> dict:
    """Resolve the customer's identity, interrupting until it succeeds.

    Args:
        state: Current graph state. The latest human message may contain a
            customer id, phone number, or email address.

    Returns:
        A partial state update setting ``customer_id``.
    """
    if state.get("customer_id"):
        return {"customer_id": state["customer_id"]}

    latest_text = ""
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            latest_text = (
                message.content if isinstance(message.content, str) else str(message.content)
            )
            break

    customer_id = _try_resolve_customer_id(latest_text)
    while not customer_id:
        customer_id = _try_resolve_customer_id(ask_for_identity())

    return {"customer_id": customer_id}
