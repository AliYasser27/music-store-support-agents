"""Tools for the Invoice Information Sub-Agent.

Every query uses SQLAlchemy bound parameters; user-supplied text is never
interpolated into SQL.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text

from src.db import engine
from src.tools._common import require_nonblank, rows_to_dicts


@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> list[dict[str, Any]]:
    """Return all invoices for a customer, most recent first.

    Args:
        customer_id: The verified Chinook CustomerId.

    Returns:
        A list of dicts with keys: InvoiceId, InvoiceDate, BillingAddress, Total.
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
        return rows_to_dicts(conn.execute(query, {"customer_id": customer_id}))


@tool
def get_invoices_sorted_by_unit_price(customer_id: str) -> list[dict[str, Any]]:
    """Return a customer's invoice line items, most expensive first.

    Args:
        customer_id: The verified Chinook CustomerId.

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
        return rows_to_dicts(conn.execute(query, {"customer_id": customer_id}))


@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> dict[str, Any]:
    """Return the support employee responsible for a given invoice.

    The customer condition is part of the query, so one customer cannot read
    another customer's support-rep details by guessing an invoice id.

    Args:
        invoice_id: The Chinook InvoiceId.
        customer_id: The verified Chinook CustomerId.

    Returns:
        A dict with keys FirstName, LastName, Email, Phone, Title, or an empty
        dict when the invoice does not belong to that customer.
    """
    invoice_id = require_nonblank(
        invoice_id, "Which invoice ID would you like the support contact for?"
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
        rows = rows_to_dicts(
            conn.execute(query, {"invoice_id": invoice_id, "customer_id": customer_id})
        )
    return rows[0] if rows else {}


INVOICE_TOOLS = [
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_employee_by_invoice_and_customer,
]
