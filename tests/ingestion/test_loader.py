"""Tests for TxtFileLoader and clean_document_text."""

import tempfile
from pathlib import Path

from legalrag.ingestion.loader import TxtFileLoader, clean_document_text

_SAMPLE_RAW = """\
CASE: 2010 BCCA 220 Harrison v. BC
YEAR: 2010
COURT: BCCA
PAGES: 3
URL: https://www.canlii.org/en/bc/bcca/doc/2010/2010bcca220/2010bcca220.html
================================================================================

--- PAGE 1 ---

First page content here. The court held that the appeal should be dismissed.

--- PAGE 2 ---

Second page content. The reasons for judgment follow.

--- PAGE 3 ---

Third page content and conclusion.
"""


class TestCleanDocumentText:
    def test_removes_header_block(self) -> None:
        cleaned = clean_document_text(_SAMPLE_RAW)
        assert "CASE:" not in cleaned
        assert "YEAR:" not in cleaned
        assert "URL:" not in cleaned
        assert "=" * 10 not in cleaned

    def test_removes_page_markers(self) -> None:
        cleaned = clean_document_text(_SAMPLE_RAW)
        assert "--- PAGE" not in cleaned

    def test_preserves_body_text(self) -> None:
        cleaned = clean_document_text(_SAMPLE_RAW)
        assert "First page content here" in cleaned
        assert "Second page content" in cleaned
        assert "Third page content" in cleaned

    def test_no_excessive_blank_lines(self) -> None:
        cleaned = clean_document_text(_SAMPLE_RAW)
        assert "\n\n\n" not in cleaned


class TestTxtFileLoader:
    def test_loads_single_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(_SAMPLE_RAW)
            tmp_path = f.name

        loader = TxtFileLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 1
        assert docs[0].text == _SAMPLE_RAW   # raw text preserved for extractor

    def test_loads_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                Path(tmpdir, f"doc{i}.txt").write_text(_SAMPLE_RAW)

            loader = TxtFileLoader()
            docs = loader.load(tmpdir)
            assert len(docs) == 3

    def test_raises_on_missing_source(self) -> None:
        import pytest

        loader = TxtFileLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/that/does/not/exist")
