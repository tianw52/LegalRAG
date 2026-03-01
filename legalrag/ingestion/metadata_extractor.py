"""Metadata extraction for CanLII-format legal documents.

All 3000 source files share a structured 5-line header:

    CASE: 2010 BCCA 220 Harrison v. British Columbia (...)
    YEAR: 2010
    COURT: BCCA
    PAGES: 11
    URL: https://www.canlii.org/...
    ================================================================================

This extractor parses the header first (high confidence), then falls back to
regex scanning of the document body for any fields still missing.

A future LLM-backed extractor can subclass BaseMetadataExtractor and swap in
without touching any other component.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from legalrag.core.interfaces import BaseMetadataExtractor
from legalrag.core.models import RawDocument

logger = logging.getLogger(__name__)

# ── Court abbreviation → full name ────────────────────────────────────────────
# Covers all courts observed in the 3000-document dataset.
COURT_ABBREV_MAP: dict[str, str] = {
    "SCC": "Supreme Court of Canada",
    "FCA": "Federal Court of Appeal",
    "FC": "Federal Court",
    "TCC": "Tax Court of Canada",
    "BCCA": "British Columbia Court of Appeal",
    "BCSC": "British Columbia Supreme Court",
    "BCPC": "British Columbia Provincial Court",
    "ONCA": "Ontario Court of Appeal",
    "ONSC": "Ontario Superior Court of Justice",
    "ONCJ": "Ontario Court of Justice",
    "ABCA": "Alberta Court of Appeal",
    "ABQB": "Alberta Court of King's Bench",
    "ABPC": "Alberta Provincial Court",
    "MBCA": "Manitoba Court of Appeal",
    "MBQB": "Manitoba Court of King's Bench",
    "SKCA": "Saskatchewan Court of Appeal",
    "SKQB": "Saskatchewan Court of King's Bench",
    "SKPC": "Saskatchewan Provincial Court",
    "NSCA": "Nova Scotia Court of Appeal",
    "NSSC": "Nova Scotia Supreme Court",
    "NSPC": "Nova Scotia Provincial Court",
    "NBBR": "New Brunswick Court of King's Bench",
    "NBCA": "New Brunswick Court of Appeal",
    "PECA": "Prince Edward Island Court of Appeal",
    "PESCTD": "Prince Edward Island Supreme Court",
    "NLCA": "Newfoundland and Labrador Court of Appeal",
    "NLSC": "Newfoundland and Labrador Supreme Court",
    "YKTC": "Yukon Territory Court",
    "NWTSC": "Northwest Territories Supreme Court",
    "NUCJ": "Nunavut Court of Justice",
    "QCCA": "Quebec Court of Appeal",
    "QCCS": "Quebec Superior Court",
    "QCCQ": "Court of Quebec",
    "QCTAQ": "Quebec Administrative Tribunal",
    # Older / numeric citation formats seen in pre-2000 files
    "CA": "Court of Appeal",
    "SC": "Superior Court",
    "PE": "Prince Edward Island",
}

# Court abbreviation → province/jurisdiction code
COURT_JURISDICTION_MAP: dict[str, str] = {
    "SCC": "federal", "FCA": "federal", "FC": "federal", "TCC": "federal",
    "BCCA": "BC", "BCSC": "BC", "BCPC": "BC",
    "ONCA": "ON", "ONSC": "ON", "ONCJ": "ON",
    "ABCA": "AB", "ABQB": "AB", "ABPC": "AB",
    "MBCA": "MB", "MBQB": "MB",
    "SKCA": "SK", "SKQB": "SK", "SKPC": "SK",
    "NSCA": "NS", "NSSC": "NS",
    "NBCA": "NB", "NBBR": "NB",
    "QCCA": "QC", "QCCS": "QC", "QCCQ": "QC",
}

# ── Compiled patterns ─────────────────────────────────────────────────────────

# Matches the structured CanLII header (URL line is optional in some older files)
_HEADER_RE = re.compile(
    r"CASE:\s*(?P<case>.+?)\n"
    r"YEAR:\s*(?P<year>\d{4})\n"
    r"COURT:\s*(?P<court>\w+)\n"
    r"PAGES:\s*(?P<pages>\d+)\n"
    r"(?:URL:\s*(?P<url>https?://\S+))?",
)

# Citation at the start of the CASE line: "2010 BCCA 220"
_INLINE_CITATION_RE = re.compile(r"^(\d{4}\s+[A-Z]+\s+\d+)\s*")

# Fallback: citation anywhere in the body text
_BODY_CITATION_RE = re.compile(
    r"(?:\[\d{4}\]\s+\w+\s+\d+|\d{4}\s+[A-Z]{2,8}\s+\d+|\d+\s+\w+\.?\d*d?\s+\d+)"
)

# Fallback: court name mentioned in body text
_BODY_COURT_RE = re.compile(
    r"(?:Supreme Court|Court of Appeal|Federal Court|High Court|"
    r"Superior Court|Provincial Court|Court of Queen's Bench|"
    r"Court of King's Bench)[^,\n]{0,60}",
    re.IGNORECASE,
)

# Fallback: date in body text
_BODY_DATE_RE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4}",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date(raw: str) -> date | None:
    raw = raw.strip().rstrip(",")
    try:
        if "-" in raw and len(raw) == 10:
            return date.fromisoformat(raw)
        if "/" in raw:
            d, m, y = raw.split("/")
            return date(int(y), int(m), int(d))
        parts = raw.replace(",", "").split()
        month = _MONTH_MAP.get(parts[0].lower())
        if month:
            return date(int(parts[2]), month, int(parts[1]))
    except (ValueError, IndexError):
        pass
    return None


# ── Extractor ─────────────────────────────────────────────────────────────────

class CanLIIMetadataExtractor(BaseMetadataExtractor):
    """
    Primary extractor for CanLII-format documents.

    Parsing priority:
      1. Structured 5-line header  (CASE / YEAR / COURT / PAGES / URL)
      2. Regex scan of document body  (fills any remaining gaps)
    """

    def __init__(self, scan_chars: int = 3000) -> None:
        self._scan_chars = scan_chars

    def extract(self, document: RawDocument) -> RawDocument:
        m = document.metadata

        # ── 1. Parse structured header ────────────────────────────────────────
        header_match = _HEADER_RE.search(document.text[:600])
        if header_match:
            case_line = header_match.group("case").strip()
            court_abbrev = header_match.group("court").strip()

            # Extract "YYYY ABBREV NNN" from the start of the CASE line
            citation_m = _INLINE_CITATION_RE.match(case_line)
            if citation_m:
                m.citation = citation_m.group(1).strip()
                remainder = case_line[citation_m.end():].strip()
                m.case_name = remainder.lstrip("–-– ").strip() or None
            else:
                m.case_name = case_line

            # Modern files: COURT field contains the abbreviation directly (e.g. "BCCA")
            # Old files: COURT field is a numeric CanLII ID; real court is in
            #            parentheses inside the CASE line e.g. "1973 2170 (FCA) ..."
            if court_abbrev.isdigit():
                # Extract abbreviation from "(FCA)" style token in the case line
                paren_m = re.search(r"\(([A-Z]{2,8})\)", case_line)
                court_abbrev = paren_m.group(1) if paren_m else court_abbrev

            m.court_abbrev = court_abbrev
            m.court = COURT_ABBREV_MAP.get(court_abbrev, court_abbrev)
            m.jurisdiction = COURT_JURISDICTION_MAP.get(court_abbrev)
            m.doc_type = "case"

            try:
                m.year = int(header_match.group("year"))
                m.decision_date = date(m.year, 1, 1)   # year-precision default
            except ValueError:
                pass

            try:
                m.pages = int(header_match.group("pages"))
            except ValueError:
                pass

            raw_url = header_match.group("url")
            m.url = raw_url.strip() if raw_url else None

        # ── 2. Fallback body scan for missing fields ───────────────────────────
        head = document.text[: self._scan_chars]

        if m.citation is None:
            cm = _BODY_CITATION_RE.search(head)
            if cm:
                m.citation = cm.group().strip()

        if m.court is None:
            ctm = _BODY_COURT_RE.search(head)
            if ctm:
                m.court = ctm.group().strip()

        # Refine year-only date to a specific date when possible
        if m.decision_date and m.decision_date.month == 1 and m.decision_date.day == 1:
            dm = _BODY_DATE_RE.search(head)
            if dm:
                precise = _parse_date(dm.group())
                if precise and precise.year == (m.year or 0):
                    m.decision_date = precise

        logger.debug(
            "Metadata '%s': citation=%s court=%s(%s) date=%s pages=%s",
            document.metadata.source_path,
            m.citation, m.court_abbrev, m.jurisdiction, m.decision_date, m.pages,
        )
        return document


# Backward-compat alias
RegexMetadataExtractor = CanLIIMetadataExtractor
