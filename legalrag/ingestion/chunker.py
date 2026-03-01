"""Hierarchical chunker for legal text.

Strategy
--------
Two-level hierarchy:
  - Parent chunks  : large, semantically coherent sections (e.g. ~1500 chars),
                     stored with a summary or as-is. Used for context window.
  - Child chunks   : smaller overlapping windows (chunk_size / chunk_overlap)
                     that are actually embedded and indexed.

Each child chunk carries a reference (parent_chunk_id) to its parent so the
generator can retrieve the full parent for richer context when needed.

This approach follows the "small-to-big retrieval" pattern:
  retrieve child (precise vector match) → expand to parent (full context).
"""

from __future__ import annotations

import logging
from uuid import uuid4

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseChunker
from legalrag.core.models import Chunk, RawDocument

logger = logging.getLogger(__name__)


def _split_into_sentences(text: str) -> list[str]:
    """Naïve sentence splitter on '.', '!', '?' boundaries."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Greedily merge consecutive sentences up to *max_chars*."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class HierarchicalChunker(BaseChunker):
    """
    Produces parent + child chunks.

    Parameters
    ----------
    parent_size:  approx character length of a parent chunk (default 1500)
    child_size:   approx character length of a child chunk  (default chunk_size from config)
    child_overlap: character overlap between consecutive child chunks
    """

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int | None = None,
        child_overlap: int | None = None,
    ) -> None:
        cfg = settings.retrieval
        self._parent_size = parent_size
        self._child_size = child_size or cfg.chunk_size
        self._child_overlap = child_overlap or cfg.chunk_overlap

    def chunk(self, document: RawDocument) -> list[Chunk]:
        sentences = _split_into_sentences(document.text)
        parent_texts = _merge_sentences(sentences, self._parent_size)

        all_chunks: list[Chunk] = []
        char_cursor = 0

        for parent_text in parent_texts:
            parent_id = str(uuid4())
            parent_start = document.text.find(parent_text, char_cursor)
            parent_end = parent_start + len(parent_text)

            # Parent chunk (not embedded – stored for context expansion)
            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=document.metadata.doc_id,
                parent_chunk_id=None,
                text=parent_text,
                char_start=parent_start,
                char_end=parent_end,
                metadata=document.metadata,
            )
            all_chunks.append(parent_chunk)

            # Child chunks within this parent
            children = self._sliding_window(parent_text, parent_start)
            for child in children:
                child.doc_id = document.metadata.doc_id
                child.parent_chunk_id = parent_id
                child.metadata = document.metadata
                all_chunks.append(child)

            char_cursor = parent_end

        logger.debug(
            "Chunked '%s' → %d chunks (%d parents)",
            document.metadata.source_path,
            len(all_chunks),
            len(parent_texts),
        )
        return all_chunks

    def _sliding_window(self, text: str, offset: int) -> list[Chunk]:
        """Produce overlapping child chunks from *text*."""
        chunks: list[Chunk] = []
        step = max(1, self._child_size - self._child_overlap)
        pos = 0
        while pos < len(text):
            end = min(pos + self._child_size, len(text))
            snippet = text[pos:end]
            chunks.append(
                Chunk(
                    doc_id="",          # filled by caller
                    text=snippet,
                    char_start=offset + pos,
                    char_end=offset + end,
                    metadata=None,      # type: ignore[arg-type]  # filled by caller
                )
            )
            if end == len(text):
                break
            pos += step
        return chunks
