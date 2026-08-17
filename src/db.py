"""Chinook database loader.

Loads the Chinook SQLite database entirely in-memory by downloading the
official SQL creation script and executing it against an in-memory SQLite
connection. This avoids the need for any external database setup.

NOTE: The helper function below is used exactly as specified by the
project requirements and must not have its behavior modified.
"""

from __future__ import annotations

import sqlite3

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def get_engine_for_chinook_db():
    """Download the Chinook SQL script and load it into an in-memory SQLite DB.

    Returns:
        A SQLAlchemy Engine backed by a single shared in-memory SQLite
        connection (via StaticPool) so that all sessions see the same data.
    """
    url = (
        "https://raw.githubusercontent.com/lerocha/chinook-database/"
        "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    )
    response = requests.get(url)
    sql_script = response.text
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


# Module-level singletons so every part of the app shares the same
# in-memory database instance.
engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)


def print_table_list() -> list[str]:
    """Print and return the list of tables available in the Chinook DB."""
    tables = db.get_usable_table_names()
    print("Chinook tables available:", sorted(tables))
    return sorted(tables)


if __name__ == "__main__":
    print_table_list()
