"""Unit tests for RegLab HF citation helpers (no network)."""

from __future__ import annotations

from evaluation.reglab.hf_reglab import (
    barexam_passage_citation,
    housing_statute_citation,
)


def test_barexam_passage_citation_format() -> None:
    assert barexam_passage_citation("42") == "barexam_qa:passage:42"
    assert barexam_passage_citation("0") == "barexam_qa:passage:0"


def test_housing_statute_citation_format() -> None:
    assert housing_statute_citation("Alabama", "7") == "housing_qa:statute:Alabama:7"


def test_housing_multiword_state() -> None:
    cite = housing_statute_citation("New York", "1")
    assert cite == "housing_qa:statute:New York:1"
    assert cite.startswith("housing_qa:statute:")
