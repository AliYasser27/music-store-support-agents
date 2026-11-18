"""Tests for the music catalog and invoice SQL tools.

These exercise real SQL against the in-memory Chinook database and need no LLM
or API key.
"""

from __future__ import annotations

from src.tools.invoice_tools import (
    get_employee_by_invoice_and_customer,
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
)
from src.tools.music_tools import (
    check_for_songs,
    get_albums_by_artist,
    get_songs_by_genre,
    get_tracks_by_artist,
)

CUSTOMER_WITH_INVOICES = "1"


def test_get_albums_by_artist_rolling_stones() -> None:
    albums = get_albums_by_artist.invoke({"artist": "Rolling Stones"})

    assert albums
    titles = {album["Title"] for album in albums}
    assert any("Voodoo Lounge" in title or "Hot Rocks" in title for title in titles)


def test_get_albums_by_artist_is_case_insensitive() -> None:
    upper = get_albums_by_artist.invoke({"artist": "ROLLING STONES"})
    lower = get_albums_by_artist.invoke({"artist": "rolling stones"})

    assert upper and upper == lower


def test_get_tracks_by_artist_returns_rows() -> None:
    tracks = get_tracks_by_artist.invoke({"artist": "Rolling Stones"})

    assert tracks
    assert {"ArtistName", "AlbumTitle", "TrackName"} <= tracks[0].keys()


def test_get_tracks_by_artist_falls_back_to_partial_name() -> None:
    """A name that is not a substring still matches on its significant word."""
    tracks = get_tracks_by_artist.invoke({"artist": "Stones Rolling"})

    assert tracks
    assert all(
        "Stones" in track["ArtistName"] or "Stones" in track["AlbumTitle"] for track in tracks
    )


def test_get_songs_by_genre_rock() -> None:
    songs = get_songs_by_genre.invoke({"genre": "Rock"})

    assert songs
    assert all("Rock" in song["GenreName"] for song in songs)


def test_check_for_songs_partial_match() -> None:
    results = check_for_songs.invoke({"song_title": "Love"})

    assert results
    assert all("love" in row["TrackName"].lower() for row in results)


def _invoices_for(customer_id: str = CUSTOMER_WITH_INVOICES) -> list[dict]:
    return get_invoices_by_customer_sorted_by_date.invoke({"customer_id": customer_id})


def test_get_invoices_by_customer_sorted_by_date() -> None:
    invoices = _invoices_for()

    assert invoices
    dates = [row["InvoiceDate"] for row in invoices]
    assert dates == sorted(dates, reverse=True)


def test_get_invoices_sorted_by_unit_price() -> None:
    lines = get_invoices_sorted_by_unit_price.invoke({"customer_id": CUSTOMER_WITH_INVOICES})

    assert lines
    prices = [float(row["UnitPrice"]) for row in lines]
    assert prices == sorted(prices, reverse=True)


def test_get_employee_by_invoice_and_customer() -> None:
    invoices = _invoices_for()
    invoice_id = str(invoices[0]["InvoiceId"])

    employee = get_employee_by_invoice_and_customer.invoke(
        {"invoice_id": invoice_id, "customer_id": CUSTOMER_WITH_INVOICES}
    )

    assert employee["FirstName"]
    assert "@" in employee["Email"]


def test_employee_lookup_rejects_another_customers_invoice() -> None:
    """An invoice id alone must not leak data across customers."""
    invoices = _invoices_for()
    invoice_id = str(invoices[0]["InvoiceId"])

    assert (
        get_employee_by_invoice_and_customer.invoke(
            {"invoice_id": invoice_id, "customer_id": "999999"}
        )
        == {}
    )


def test_artist_name_cannot_inject_sql() -> None:
    """A quote in the search term is treated as data, not SQL."""
    assert get_albums_by_artist.invoke({"artist": "'; DROP TABLE Album; --"}) == []
    assert get_albums_by_artist.invoke({"artist": "Rolling Stones"})
