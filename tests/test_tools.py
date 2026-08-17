"""Tests for the music catalog and invoice SQL tools.

These tests only exercise SQL/tool logic against the real Chinook data
(loaded in-memory over the network) -- they do not require an LLM or API
key.
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


def test_get_albums_by_artist_rolling_stones() -> None:
    albums = get_albums_by_artist.invoke({"artist": "Rolling Stones"})
    assert isinstance(albums, list)
    assert len(albums) > 0
    titles = {a["Title"] for a in albums}
    assert any("Voodoo Lounge" in t or "Hot Rocks" in t or "No Security" in t for t in titles)


def test_get_tracks_by_artist_returns_rows() -> None:
    tracks = get_tracks_by_artist.invoke({"artist": "Rolling Stones"})
    assert isinstance(tracks, list)
    assert len(tracks) > 0
    assert "TrackName" in tracks[0]


def test_get_songs_by_genre_rock() -> None:
    songs = get_songs_by_genre.invoke({"genre": "Rock"})
    assert isinstance(songs, list)
    assert len(songs) > 0


def test_check_for_songs_partial_match() -> None:
    results = check_for_songs.invoke({"song_title": "Love"})
    assert isinstance(results, list)
    assert len(results) > 0


def test_get_invoices_by_customer_sorted_by_date() -> None:
    invoices = get_invoices_by_customer_sorted_by_date.invoke({"customer_id": "1"})
    assert isinstance(invoices, list)
    if len(invoices) > 1:
        dates = [row["InvoiceDate"] for row in invoices]
        assert dates == sorted(dates, reverse=True)


def test_get_invoices_sorted_by_unit_price() -> None:
    lines = get_invoices_sorted_by_unit_price.invoke({"customer_id": "1"})
    assert isinstance(lines, list)
    if len(lines) > 1:
        prices = [float(row["UnitPrice"]) for row in lines]
        assert prices == sorted(prices, reverse=True)


def test_get_employee_by_invoice_and_customer() -> None:
    invoices = get_invoices_by_customer_sorted_by_date.invoke({"customer_id": "1"})
    assert invoices, "expected customer 1 to have at least one invoice"
    invoice_id = str(invoices[0]["InvoiceId"])
    employee = get_employee_by_invoice_and_customer.invoke(
        {"invoice_id": invoice_id, "customer_id": "1"}
    )
    assert isinstance(employee, dict)
    if employee:
        assert "FirstName" in employee and "Email" in employee
