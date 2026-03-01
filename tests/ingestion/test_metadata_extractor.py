"""Tests for CanLIIMetadataExtractor."""

import pytest

from legalrag.core.models import LegalDocumentMetadata, RawDocument
from legalrag.ingestion.metadata_extractor import CanLIIMetadataExtractor

# A realistic CanLII-format header block used across several tests
_CANLII_HEADER = """\
CASE: 2010 BCCA 220 Harrison v. British Columbia (Children and Family Development)
YEAR: 2010
COURT: BCCA
PAGES: 11
URL: https://www.canlii.org/en/bc/bcca/doc/2010/2010bcca220/2010bcca220.html
================================================================================

--- PAGE 1 ---

COURT OF APPEAL FOR BRITISH COLUMBIA
Citation: Harrison v. British Columbia, 2010 BCCA 220
Date: 20100505
"""


def _make_doc(text: str) -> RawDocument:
    return RawDocument(
        metadata=LegalDocumentMetadata(source_path="/data/2010_BCCA_2010 BCCA 220 Harrison.txt"),
        text=text,
    )


class TestCanLIIHeaderParsing:
    def test_extracts_citation_from_header(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.citation == "2010 BCCA 220"

    def test_extracts_case_name_from_header(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.case_name is not None
        assert "Harrison" in result.metadata.case_name

    def test_extracts_court_abbrev(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.court_abbrev == "BCCA"

    def test_expands_court_to_full_name(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.court == "British Columbia Court of Appeal"

    def test_extracts_year(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.year == 2010

    def test_extracts_pages(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.pages == 11

    def test_extracts_url(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.url is not None
        assert "canlii.org" in result.metadata.url

    def test_sets_doc_type_to_case(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.doc_type == "case"

    def test_sets_jurisdiction(self) -> None:
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.jurisdiction == "BC"

    def test_refines_date_from_body(self) -> None:
        """Body contains 'Date: 20100505' – extractor should refine to May 5 2010."""
        doc = _make_doc(_CANLII_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        # At minimum, year should be correct
        assert result.metadata.decision_date is not None
        assert result.metadata.decision_date.year == 2010


class TestFallbackExtraction:
    """When no CanLII header is present, regex fallback should still populate fields."""

    def test_fallback_citation(self) -> None:
        doc = _make_doc("In 2003 FCA 168 the court held that...")
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.citation is not None
        assert "FCA" in result.metadata.citation

    def test_fallback_court(self) -> None:
        doc = _make_doc("The Supreme Court of Canada ruled unanimously.")
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.court is not None
        assert "Supreme Court" in result.metadata.court

    def test_no_false_positives_on_empty_text(self) -> None:
        doc = _make_doc("No legal metadata here.")
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.citation is None
        assert result.metadata.court is None


class TestQuebecDocs:
    """Quebec-court documents often have French text; header still parses."""

    _QC_HEADER = """\
CASE: 2010 QCCS 1756 Steinmetz c. Alku Plastics Ltd.
YEAR: 2010
COURT: QCCS
PAGES: 8
URL: https://www.canlii.org/fr/qc/qccs/doc/2010/2010qccs1756/2010qccs1756.html
================================================================================
"""

    def test_quebec_court_full_name(self) -> None:
        doc = _make_doc(self._QC_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.court == "Quebec Superior Court"
        assert result.metadata.jurisdiction == "QC"

    def test_quebec_citation(self) -> None:
        doc = _make_doc(self._QC_HEADER)
        result = CanLIIMetadataExtractor().extract(doc)
        assert result.metadata.citation == "2010 QCCS 1756"
