"""Customer verification node.

Parses the latest user message for a customer_id, email, or phone number,
resolves it to a Chinook CustomerId, and stores it in short-term state. If
no valid identity can be resolved, the node interrupts and asks the user to
supply one, looping until verification succeeds.
"""

from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import text

from src.db import engine
from src.nodes.hitl import ask_for_identity
from src.state import State

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CUSTOMER_ID_RE = re.compile(r"\bcustomer\s*id\D{0,5}(\d+)\b", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^\s*(\d{1,10})\s*$")
_PHONE_RE = re.compile(r"[+()\d][\d()+\-.\s]{5,}\d")


def _normalize_phone(raw: str) -> str:
    """Strip spaces, parentheses, dashes, and a leading '+' from a phone string."""
    digits = re.sub(r"[\s()\-]", "", raw)
    digits = digits.lstrip("+")
    return digits


def _lookup_by_customer_id(customer_id: str) -> Optional[str]:
    query = text("SELECT CustomerId FROM Customer WHERE CustomerId = :cid")
    with engine.connect() as conn:
        row = conn.execute(query, {"cid": customer_id}).fetchone()
        return str(row[0]) if row else None


def _lookup_by_email(email: str) -> Optional[str]:
    query = text("SELECT CustomerId FROM Customer WHERE LOWER(Email) = LOWER(:email)")
    with engine.connect() as conn:
        row = conn.execute(query, {"email": email}).fetchone()
        return str(row[0]) if row else None


def _lookup_by_phone(phone: str) -> Optional[str]:
    """Match a normalized phone number against normalized Customer.Phone values."""
    normalized_input = _normalize_phone(phone)
    if not normalized_input:
        return None
    query = text("SELECT CustomerId, Phone FROM Customer WHERE Phone IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    for customer_id, db_phone in rows:
        if _normalize_phone(db_phone) == normalized_input:
            return str(customer_id)
        # Also allow suffix match in case of differing country-code prefixes.
        if normalized_input.endswith(_normalize_phone(db_phone)[-9:]) and len(
            _normalize_phone(db_phone)
        ) >= 9:
            return str(customer_id)
    return None


def _try_resolve_customer_id(user_text: str) -> Optional[str]:
    """Attempt to resolve a Chinook CustomerId from free-form user text."""
    if not user_text:
        return None

    match = _CUSTOMER_ID_RE.search(user_text)
    if match:
        resolved = _lookup_by_customer_id(match.group(1))
        if resolved:
            return resolved

    bare_match = _BARE_ID_RE.match(user_text)
    if bare_match:
        resolved = _lookup_by_customer_id(bare_match.group(1))
        if resolved:
            return resolved

    email_match = _EMAIL_RE.search(user_text)
    if email_match:
        resolved = _lookup_by_email(email_match.group(0))
        if resolved:
            return resolved

    phone_match = _PHONE_RE.search(user_text)
    if phone_match:
        resolved = _lookup_by_phone(phone_match.group(0))
        if resolved:
            return resolved

    return None


def verify_customer_node(state: State) -> dict:
    """Resolve and verify the customer's identity, interrupting if needed.

    Args:
        state: Current graph state. Expects the latest HumanMessage to
            possibly contain a customer_id, phone, or email.

    Returns:
        A partial state update setting ``customer_id``.
    """
    if state.get("customer_id"):
        return {"customer_id": state["customer_id"]}

    latest_text = ""
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        if role == "human":
            latest_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    customer_id = _try_resolve_customer_id(latest_text)
    while not customer_id:
        user_reply = ask_for_identity()
        customer_id = _try_resolve_customer_id(user_reply)

    return {"customer_id": customer_id}
