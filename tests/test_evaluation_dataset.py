"""Validation for the labelled routing set.

These tests check the evaluation data itself, so a malformed or unrepresentative
case file fails CI without needing an API key.
"""

from __future__ import annotations

from collections import Counter

from evaluation.routing_eval import ROUTES, load_cases

CATALOG_WORDS = {
    "album",
    "albums",
    "artist",
    "artists",
    "band",
    "catalog",
    "genre",
    "genres",
    "music",
    "recommend",
    "song",
    "songs",
    "track",
    "tracks",
}
ACCOUNT_WORDS = {
    "bought",
    "buy",
    "cost",
    "invoice",
    "invoices",
    "order",
    "orders",
    "paid",
    "pay",
    "purchase",
    "purchased",
    "purchases",
    "receipt",
    "refund",
}


def _words(message: str) -> set[str]:
    return {word.strip(".,?!'\"()").lower() for word in message.split()}


def test_cases_load_and_are_unique() -> None:
    cases = load_cases()

    assert len(cases) >= 20
    assert len({case.message for case in cases}) == len(cases)


def test_every_route_is_represented() -> None:
    counts = Counter(case.expected for case in load_cases())

    assert set(counts) == set(ROUTES)
    assert min(counts.values()) >= 3, f"under-represented route: {counts}"
    assert counts["both"] >= 5, "the expensive route needs more than a token case"


def test_every_case_documents_its_intent() -> None:
    assert all(len(case.note) >= 10 for case in load_cases())


def test_both_cases_touch_both_domains() -> None:
    """A `both` label is only defensible when the message spans both domains."""
    for case in load_cases():
        if case.expected != "both":
            continue
        words = _words(case.message)
        assert words & CATALOG_WORDS, f"no catalog intent in: {case.message}"
        assert words & ACCOUNT_WORDS, f"no account intent in: {case.message}"


def test_single_domain_cases_stay_in_one_domain() -> None:
    """Guards against mislabelling a two-intent message as a single route."""
    for case in load_cases():
        if case.expected != "music_catalog":
            continue
        assert not (_words(case.message) & ACCOUNT_WORDS), case.message
