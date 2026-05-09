"""Tests for PassthroughQueryFormulator (no LLM needed)."""

from legalrag.query.formulator import PassthroughQueryFormulator


def test_passthrough_preserves_query() -> None:
    formulator = PassthroughQueryFormulator()
    result = formulator.formulate("What is reasonable doubt?")
    assert result.raw_query == "What is reasonable doubt?"
    assert result.reformulated_query == "What is reasonable doubt?"


def test_passthrough_empty_filters() -> None:
    formulator = PassthroughQueryFormulator()
    result = formulator.formulate("test")
    assert result.court_filter is None
    assert result.citation_filter is None
    assert result.date_from is None
