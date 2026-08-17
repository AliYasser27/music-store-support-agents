"""Tools for the Music Catalog Sub-Agent.

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


def _require_nonblank(value: str, ask: str) -> str:
    """Ensure a required tool argument is non-blank, interrupting if not.

    If ``value`` is blank (empty/whitespace), this pauses the graph via
    ``interrupt()`` and asks the user for the missing detail. When the
    graph is resumed, the user's reply is used in place of the original
    (missing) value.

    Args:
        value: The argument value supplied by the calling agent/LLM.
        ask: The clarification question to show the user if value is blank.

    Returns:
        The original value if non-blank, otherwise the user's resumed reply.
    """
    if value and value.strip():
        return value.strip()
    reply = interrupt({"question": ask})
    return str(reply).strip()


def _rows_to_dicts(result) -> list[dict[str, Any]]:
    """Convert a SQLAlchemy CursorResult into a list of plain dicts."""
    return [dict(row._mapping) for row in result]


@tool
def get_albums_by_artist(artist: str) -> list[dict[str, Any]]:
    """Return albums released by the given artist.

    Args:
        artist: The artist's name (partial, case-insensitive match allowed).

    Returns:
        A list of dicts with keys: ArtistName, AlbumId, Title.
    """
    artist = _require_nonblank(
        artist, "Which artist would you like albums for?"
    )
    query = text(
        """
        SELECT ar.Name AS ArtistName, al.AlbumId AS AlbumId, al.Title AS Title
        FROM Artist ar
        JOIN Album al ON al.ArtistId = ar.ArtistId
        WHERE ar.Name LIKE :artist
        ORDER BY al.Title
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"artist": f"%{artist}%"})
        return _rows_to_dicts(result)


@tool
def get_tracks_by_artist(artist: str) -> list[dict[str, Any]]:
    """Return tracks recorded by the given artist.

    If no exact/partial match is found for the artist, this falls back to
    a broader search across similarly-named artists.

    Args:
        artist: The artist's name (partial, case-insensitive match allowed).

    Returns:
        A list of dicts with keys: ArtistName, AlbumTitle, TrackName.
    """
    artist = _require_nonblank(
        artist, "Which artist would you like tracks for?"
    )
    query = text(
        """
        SELECT ar.Name AS ArtistName, al.Title AS AlbumTitle, t.Name AS TrackName
        FROM Artist ar
        JOIN Album al ON al.ArtistId = ar.ArtistId
        JOIN Track t ON t.AlbumId = al.AlbumId
        WHERE ar.Name LIKE :artist
        ORDER BY al.Title, t.TrackId
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"artist": f"%{artist}%"})
        rows = _rows_to_dicts(result)

    if rows:
        return rows

    # Fallback: try splitting the artist name into words and matching any of them.
    words = [w for w in artist.split() if len(w) > 2]
    if not words:
        return []

    fallback_query = text(
        """
        SELECT ar.Name AS ArtistName, al.Title AS AlbumTitle, t.Name AS TrackName
        FROM Artist ar
        JOIN Album al ON al.ArtistId = ar.ArtistId
        JOIN Track t ON t.AlbumId = al.AlbumId
        WHERE ar.Name LIKE :word
        ORDER BY al.Title, t.TrackId
        LIMIT 50
        """
    )
    with engine.connect() as conn:
        result = conn.execute(fallback_query, {"word": f"%{words[0]}%"})
        return _rows_to_dicts(result)


@tool
def get_songs_by_genre(genre: str) -> list[dict[str, Any]]:
    """Return songs that belong to the given genre.

    Args:
        genre: The genre name (partial, case-insensitive match allowed),
            e.g. "Rock", "Jazz", "Metal".

    Returns:
        A list of dicts with keys: GenreName, ArtistName, TrackName.
    """
    genre = _require_nonblank(
        genre, "Which genre are you interested in?"
    )
    query = text(
        """
        SELECT g.Name AS GenreName, ar.Name AS ArtistName, t.Name AS TrackName
        FROM Genre g
        JOIN Track t ON t.GenreId = g.GenreId
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE g.Name LIKE :genre
        ORDER BY ar.Name, t.Name
        LIMIT 100
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"genre": f"%{genre}%"})
        return _rows_to_dicts(result)


@tool
def check_for_songs(song_title: str) -> list[dict[str, Any]]:
    """Check whether a song with the given (partial) title exists.

    Args:
        song_title: The song title, or a fragment of it, to search for.
            Matching is case-insensitive and allows partial matches.

    Returns:
        A list of dicts with keys: TrackName, ArtistName, AlbumTitle.
    """
    song_title = _require_nonblank(
        song_title, "What is the song title (or part of it) you're looking for?"
    )
    query = text(
        """
        SELECT t.Name AS TrackName, ar.Name AS ArtistName, al.Title AS AlbumTitle
        FROM Track t
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE t.Name LIKE :song_title
        ORDER BY t.Name
        LIMIT 50
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"song_title": f"%{song_title}%"})
        return _rows_to_dicts(result)


MUSIC_TOOLS = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
]
