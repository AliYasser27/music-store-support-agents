"""Tools for the Music Catalog Sub-Agent.

Every query uses SQLAlchemy bound parameters; user-supplied text is never
interpolated into SQL.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text

from src.db import engine
from src.tools._common import require_nonblank, rows_to_dicts

_TRACKS_BY_ARTIST_SQL = """
    SELECT ar.Name AS ArtistName, al.Title AS AlbumTitle, t.Name AS TrackName
    FROM Artist ar
    JOIN Album al ON al.ArtistId = ar.ArtistId
    JOIN Track t ON t.AlbumId = al.AlbumId
    WHERE ar.Name LIKE :artist
    ORDER BY al.Title, t.TrackId
"""

_FALLBACK_TRACKS_BY_ARTIST_SQL = """
    SELECT ar.Name AS ArtistName, al.Title AS AlbumTitle, t.Name AS TrackName
    FROM Artist ar
    JOIN Album al ON al.ArtistId = ar.ArtistId
    JOIN Track t ON t.AlbumId = al.AlbumId
    WHERE ar.Name LIKE :word
    ORDER BY al.Title, t.TrackId
    LIMIT 50
"""


@tool
def get_albums_by_artist(artist: str) -> list[dict[str, Any]]:
    """Return albums released by the given artist.

    Args:
        artist: The artist's name (case-insensitive partial match).

    Returns:
        A list of dicts with keys: ArtistName, AlbumId, Title.
    """
    artist = require_nonblank(artist, "Which artist would you like albums for?")
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
        return rows_to_dicts(conn.execute(query, {"artist": f"%{artist}%"}))


@tool
def get_tracks_by_artist(artist: str) -> list[dict[str, Any]]:
    """Return tracks recorded by the given artist.

    If the artist name does not match anything, the search widens to the
    first significant word of the name, which tolerates spelling variants
    such as "Rolling Stones" vs. "The Rolling Stones".

    Args:
        artist: The artist's name (case-insensitive partial match).

    Returns:
        A list of dicts with keys: ArtistName, AlbumTitle, TrackName.
    """
    artist = require_nonblank(artist, "Which artist would you like tracks for?")
    with engine.connect() as conn:
        rows = rows_to_dicts(conn.execute(text(_TRACKS_BY_ARTIST_SQL), {"artist": f"%{artist}%"}))
    if rows:
        return rows

    words = [word for word in artist.split() if len(word) > 2]
    if not words:
        return []

    with engine.connect() as conn:
        return rows_to_dicts(
            conn.execute(text(_FALLBACK_TRACKS_BY_ARTIST_SQL), {"word": f"%{words[0]}%"})
        )


@tool
def get_songs_by_genre(genre: str) -> list[dict[str, Any]]:
    """Return songs that belong to the given genre.

    Args:
        genre: The genre name (case-insensitive partial match), e.g. "Rock".

    Returns:
        A list of dicts with keys: GenreName, ArtistName, TrackName.
    """
    genre = require_nonblank(genre, "Which genre are you interested in?")
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
        return rows_to_dicts(conn.execute(query, {"genre": f"%{genre}%"}))


@tool
def check_for_songs(song_title: str) -> list[dict[str, Any]]:
    """Check whether a song with the given partial title exists.

    Args:
        song_title: The song title, or a fragment of it (case-insensitive).

    Returns:
        A list of dicts with keys: TrackName, ArtistName, AlbumTitle.
    """
    song_title = require_nonblank(
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
        return rows_to_dicts(conn.execute(query, {"song_title": f"%{song_title}%"}))


MUSIC_TOOLS = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
]
