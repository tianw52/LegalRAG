"""Plain-text document loader for CanLII-format legal documents.

Each file has a structured 5-line header followed by a page-separated body:

    CASE: 2010 BCCA 220 ...
    YEAR: 2010
    COURT: BCCA
    PAGES: 11
    URL: https://...
    ================================================================================

    --- PAGE 1 ---
    ...text...
    --- PAGE 2 ---
    ...

The loader returns the **raw** text unchanged so the metadata extractor can
read the header.  The pipeline calls ``clean_document_text()`` afterwards to
strip the header block and page markers before chunking.

Extend by subclassing BaseLoader for other formats (PDF, DOCX, HTML, …).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from legalrag.core.interfaces import BaseLoader
from legalrag.core.models import LegalDocumentMetadata, RawDocument

logger = logging.getLogger(__name__)

# ── Text cleaning helpers (called by IngestionPipeline after metadata extraction)

# Matches the CanLII header block up to and including the separator line
_HEADER_BLOCK_RE = re.compile(
    r"^CASE:.*?^={10,}\s*",
    re.DOTALL | re.MULTILINE,
)

# Matches page separator markers: "--- PAGE 1 ---"
_PAGE_SEP_RE = re.compile(r"^---\s*PAGE\s*\d+\s*---\s*$", re.MULTILINE)

# Collapses 3+ consecutive blank lines into 2
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


def clean_document_text(raw: str) -> str:
    """Strip CanLII header block and page markers; collapse excess whitespace."""
    text = _HEADER_BLOCK_RE.sub("", raw, count=1)
    text = _PAGE_SEP_RE.sub("\n", text)
    text = _EXCESS_BLANK_RE.sub("\n\n", text)
    return text.strip()


class TxtFileLoader(BaseLoader):
    """Load one or many CanLII plain-text files into RawDocument objects.

    Returns raw file text (including the structured header) so that the
    metadata extractor downstream can parse the header fields.
    ``clean_document_text()`` should be called after metadata extraction.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def load(self, source: str) -> list[RawDocument]:
        path = Path(source)
        try:
            is_file = path.is_file()
            is_dir = path.is_dir()
        except PermissionError as exc:
            raise FileNotFoundError(f"Source not accessible: {source}") from exc
        if is_file:
            return [self._load_file(path)]
        if is_dir:
            files = sorted(path.rglob("*.txt"))
            logger.info("Found %d .txt files in %s", len(files), path)
            return [self._load_file(f) for f in files]
        raise FileNotFoundError(f"Source not found: {source}")

    def _load_file(self, path: Path) -> RawDocument:
        raw = path.read_text(encoding=self._encoding)
        metadata = LegalDocumentMetadata(source_path=str(path))
        logger.debug("Loaded %s (%d chars)", path.name, len(raw))
        return RawDocument(metadata=metadata, text=raw)
