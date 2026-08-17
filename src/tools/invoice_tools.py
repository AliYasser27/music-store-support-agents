"""Tools for the Invoice Information Sub-Agent.

All tools query the Chinook database via SQLAlchemy using bound
parameters only -- user-supplied strings are NEVER concatenated directly
into SQL text.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt
from sqlalchemy import text

from src.db import engine


def _rows_to_dicts(result) -> list[dict[str, Any]]:
    """Convert a SQLAlchemy CursorResult into a list of plain dicts."""
    return [dict(row._mapping) for row in result]


def _require_nonblank(value: str, ask: str) -> str:
    """Ensure a required tool argument is non-blank, interrupting if not.

    See ``music_tools._require_nonblank`` for the identical pattern used
    here for the Invoice Information Sub-Agent's tools.
    """
    if value and str(value).strip():
        return str(value).strip()
    reply = interrupt({"question": ask})
    return str(reply).strip()


@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> list[dict[str, Any]]:
    """Return all invoices for a customer, sorted by InvoiceDate descending.

    Args:
        customer_id: The Chinook CustomerId (as a string or int-like value).

    Returns:
        A list of dicts with keys: InvoiceId, InvoiceDate, BillingAddress,
        Total.
    """
    query = text(
        """
        SELECT InvoiceId, InvoiceDate, BillingAddress, Total
        FROM Invoice
        WHERE CustomerId = :customer_id
        ORDER BY InvoiceDate DESC
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"customer_id": customer_id})
        return _rows_to_dicts(result)


@tool
def get_invoices_sorted_by_unit_price(customer_id: str) -> list[dict[str, Any]]:
    """Return all invoice line items for a customer, sorted by UnitPrice descending.

    Args:
        customer_id: The Chinook CustomerId (as a string or int-like value).

    Returns:
        A list of dicts with keys: InvoiceId, TrackId, UnitPrice, Quantity.
    """
    query = text(
        """
        SELECT il.InvoiceId AS InvoiceId, il.TrackId AS TrackId,
               il.UnitPrice AS UnitPrice, il.Quantity AS Quantity
        FROM InvoiceLine il
        JOIN Invoice i ON i.InvoiceId = il.InvoiceId
        WHERE i.CustomerId = :customer_id
        ORDER BY il.UnitPrice DESC
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"customer_id": customer_id})
        return _rows_to_dicts(result)


@tool
def get_employee_by_invoice_and_customer(
    invoice_id: str, customer_id: str
) -> dict[str, Any]:
    """Return the support employee associated with a given invoice and customer.

    Traverses Customer -> SupportRepId -> Employee, after confirming the
    invoice belongs to the given customer.

    Args:
        invoice_id: The Chinook InvoiceId.
        customer_id: The Chinook CustomerId.

    Returns:
        A dict with keys: FirstName, LastName, Email, Phone, Title. Empty
        dict if no matching employee/invoice/customer combination exists.
    """
    invoice_id = _require_nonblank(
        invoice_id,
        "Which invoice ID would you like the support contact for?",
    )
    query = text(
        """
        SELECT e.FirstName AS FirstName, e.LastName AS LastName,
               e.Email AS Email, e.Phone AS Phone, e.Title AS Title
        FROM Invoice i
        JOIN Customer c ON c.CustomerId = i.CustomerId
        JOIN Employee e ON e.EmployeeId = c.SupportRepId
        WHERE i.InvoiceId = :invoice_id AND i.CustomerId = :customer_id
        """
    )
    with engine.connect() as conn:
        result = conn.execute(
            query, {"invoice_id": invoice_id, "customer_id": customer_id}
        )
        rows = _rows_to_dicts(result)
        return rows[0] if rows else {}


INVOICE_TOOLS = [
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_employee_by_invoice_and_customer,
]
