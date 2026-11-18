"""Chinook database access.

The Chinook sample database is downloaded as a SQL script and loaded into an
in-memory SQLite connection, so the project needs no database server and no
setup step. The module exposes one shared ``engine`` used by every tool and
node; keeping a single engine is what makes the in-memory data visible
everywhere.
"""

from __future__ import annotations

import sqlite3

import requests
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

from src.config import get_settings


class DatabaseLoadError(RuntimeError):
    """Raised when the Chinook script cannot be downloaded or executed."""


def _download_chinook_sql() -> str:
    """Fetch the Chinook DDL/DML script used to populate the database."""
    settings = get_settings()
    try:
        response = requests.get(settings.chinook_sql_url, timeout=settings.chinook_http_timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatabaseLoadError(
            "Could not download the Chinook database script from "
            f"{settings.chinook_sql_url}. The first run needs internet access. "
            f"Original error: {exc}"
        ) from exc
    return response.text


def get_engine_for_chinook_db():
    """Build an in-memory SQLite engine populated with the Chinook dataset.

    Returns:
        A SQLAlchemy engine backed by one shared in-memory connection (via
        ``StaticPool``), so every session sees the same data.
    """
    sql_script = _download_chinook_sql()
    try:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(sql_script)
    except sqlite3.Error as exc:
        raise DatabaseLoadError(f"Failed to load the Chinook dataset: {exc}") from exc

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


# Shared singleton: every tool and node queries the same in-memory database.
engine = get_engine_for_chinook_db()


def get_table_names() -> list[str]:
    """Return the sorted list of tables available in the Chinook database."""
    return sorted(inspect(engine).get_table_names())


def print_table_list() -> list[str]:
    """Print and return the Chinook table names (used by the notebook)."""
    tables = get_table_names()
    print("Chinook tables available:", tables)
    return tables


if __name__ == "__main__":
    print_table_list()
