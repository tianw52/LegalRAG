"""Chunkers for legal text.

Available chunkers
------------------
HierarchicalChunker
    Two-level hierarchy: parent chunks (~1500 chars) + overlapping child chunks
    (~512 chars, 64-char overlap).  Child chunks are embedded and retrieved; parents are fetched at
    generation time for richer context ("small-to-big retrieval").

RecursiveCharacterTextSplitter
    Flat chunker inspired by LangChain's RecursiveCharacterTextSplitter.  Tries to
    split on progressively finer separators (paragraph → sentence → word →
    character) so that each chunk stays under *chunk_size* characters while
    preserving as much natural language structure as possible.  Consecutive chunks
    share *chunk_overlap* characters so context is not abruptly lost at boundaries.

Implementation note
-------------------
Splitting is done entirely by *character position* in the original text so
that ``char_start`` / ``char_end`` are always exact offsets into the source
document.  We never reconstruct text by joining tokens (which loses whitespace)
and never use ``str.find()`` on reconstructed strings (which is fragile when
the original contains multi-space / newline runs).
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseChunker
from legalrag.core.models import Chunk, RawDocument, stable_id

logger = logging.getLogger(__name__)


# ── Sentence-boundary positions ───────────────────────────────────────────────

_SENT_END_RE = re.compile(r"[.!?](?=\s)")


def _sentence_end_positions(text: str) -> list[int]:
    """Return character positions *just after* each sentence-ending punctuation.

    Each value is the index of the first whitespace character after '.', '!',
    or '?'.  The caller uses these as candidate split points when building
    parent chunks.
    """
    return [m.end() for m in _SENT_END_RE.finditer(text)]


# ── Parent-level splitting (position-based) ───────────────────────────────────


def _split_positions(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans that partition *text* completely.

    Spans are at most *max_chars* wide.  We try to break at a sentence boundary
    (first ``[.!?]\\s`` after the target length) to keep coherent units, but
    always fall back to a hard cut so no character is ever skipped.

    The spans are contiguous and non-overlapping: ``spans[i][1] == spans[i+1][0]``
    and the union covers ``[0, len(text))``.
    """
    if not text:
        return []

    ends = _sentence_end_positions(text)
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)

    while start < n:
        target = start + max_chars
        if target >= n:
            spans.append((start, n))
            break

        # Find the first sentence boundary at or after `target`
        # (search within a reasonable lookahead to avoid runaway chunks)
        lookahead = target + max_chars // 2
        boundary = next(
            (e for e in ends if target <= e <= lookahead),
            None,
        )
        end = boundary if boundary is not None else target
        spans.append((start, end))
        start = end

    return spans


# ── HierarchicalChunker ───────────────────────────────────────────────────────


class HierarchicalChunker(BaseChunker):
    """Produces parent + child chunks with exact character offsets.

    Parameters
    ----------
    parent_size:
        Approximate character length of a parent chunk (default 1500).
    child_size:
        Approximate character length of a child chunk (default from config).
    child_overlap:
        Character overlap between consecutive child chunks (default from config).
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

    @property
    def is_hierarchical(self) -> bool:
        return True

    def chunk(self, document: RawDocument) -> list[Chunk]:
        text = document.text
        doc_id = document.metadata.doc_id
        parent_spans = _split_positions(text, self._parent_size)

        all_chunks: list[Chunk] = []

        for p_start, p_end in parent_spans:
            parent_text = text[p_start:p_end]
            parent_id = stable_id(doc_id, str(p_start), str(p_end))

            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                parent_chunk_id=None,
                is_parent=True,
                text=parent_text,
                char_start=p_start,
                char_end=p_end,
                metadata=document.metadata,
            )
            all_chunks.append(parent_chunk)

            # Child chunks: sliding window over the parent span
            children = self._sliding_window(
                parent_text, offset=p_start, doc_id=doc_id, parent_id=parent_id,
                metadata=document.metadata,
            )
            all_chunks.extend(children)

        logger.debug(
            "Chunked '%s' → %d chunks (%d parents)",
            document.metadata.source_path,
            len(all_chunks),
            len(parent_spans),
        )
        return all_chunks

    def _sliding_window(
        self,
        text: str,
        offset: int,
        doc_id: str,
        parent_id: str,
        metadata,
    ) -> list[Chunk]:
        """Produce overlapping child chunks with deterministic IDs."""
        chunks: list[Chunk] = []
        step = max(1, self._child_size - self._child_overlap)
        pos = 0
        n = len(text)

        while pos < n:
            end = min(pos + self._child_size, n)
            char_start = offset + pos
            char_end = offset + end
            chunks.append(
                Chunk(
                    chunk_id=stable_id(doc_id, str(char_start), str(char_end)),
                    doc_id=doc_id,
                    parent_chunk_id=parent_id,
                    text=text[pos:end],
                    char_start=char_start,
                    char_end=char_end,
                    metadata=metadata,
                )
            )
            if end == n:
                break
            pos += step

        return chunks


# ── RecursiveCharacterTextSplitter ────────────────────────────────────────────

#: Default separator ladder: paragraph → sentence end → whitespace → any char.
_DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence boundary (period + space)
    "! ",
    "? ",
    "; ",
    ", ",
    " ",      # word boundary
    "",       # character-level fallback
)


class RecursiveCharacterTextSplitter(BaseChunker):
    """Flat chunker that splits on progressively finer separators.

    The algorithm mirrors LangChain's ``RecursiveCharacterTextSplitter`` but
    operates on exact character offsets so ``char_start`` / ``char_end`` are
    always consistent with the original document text.

    All produced chunks have ``parent_chunk_id=None`` (flat, no hierarchy).
    Consecutive chunks share *chunk_overlap* characters for context continuity.

    Parameters
    ----------
    chunk_size:
        Maximum character length of a single chunk (default from config).
    chunk_overlap:
        Number of characters shared between consecutive chunks (default from
        config).
    separators:
        Ordered sequence of separator strings tried from coarsest to finest.
        The splitter uses the first separator that keeps all pieces ≤
        *chunk_size*; if none works, it falls back to the next until the
        character-level fallback (empty string) is reached.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: Sequence[str] = _DEFAULT_SEPARATORS,
    ) -> None:
        cfg = settings.retrieval
        self._chunk_size = chunk_size or cfg.chunk_size
        self._chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self._separators = list(separators)

    # ── public API ────────────────────────────────────────────────────────────

    def chunk(self, document: RawDocument) -> list[Chunk]:
        text = document.text
        doc_id = document.metadata.doc_id

        # Collect (start, end) spans via the recursive split
        spans = self._split_text(text, separators=self._separators)

        # Merge short spans and apply overlap to build final windows
        windows = self._merge_with_overlap(text, spans)

        chunks: list[Chunk] = []
        for char_start, char_end in windows:
            chunks.append(
                Chunk(
                    chunk_id=stable_id(doc_id, str(char_start), str(char_end)),
                    doc_id=doc_id,
                    parent_chunk_id=None,
                    text=text[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    metadata=document.metadata,
                )
            )

        logger.debug(
            "RecursiveCharacterTextSplitter: '%s' → %d chunks",
            document.metadata.source_path,
            len(chunks),
        )
        return chunks

    # ── internals ─────────────────────────────────────────────────────────────

    def _split_text(
        self, text: str, separators: list[str]
    ) -> list[tuple[int, int]]:
        """Recursively split *text* into spans ≤ chunk_size using *separators*.

        Returns a flat list of ``(start, end)`` spans (absolute offsets into
        *text*) that together cover ``[0, len(text))``.
        """
        if not text:
            return []

        # If text already fits, return as-is
        if len(text) <= self._chunk_size:
            return [(0, len(text))]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level fallback: hard-cut at chunk_size
            return [(i, min(i + self._chunk_size, len(text)))
                    for i in range(0, len(text), self._chunk_size)]

        # Split on this separator; keep the separator at the *end* of each
        # piece so sentence punctuation stays with its sentence.
        pattern = re.escape(sep)
        pieces: list[tuple[int, int]] = []
        prev = 0
        for m in re.finditer(pattern, text):
            end = m.end()
            pieces.append((prev, end))
            prev = end
        if prev < len(text):
            pieces.append((prev, len(text)))

        # If every piece fits, we're done
        if all(end - start <= self._chunk_size for start, end in pieces):
            return pieces

        # Otherwise recurse on pieces that are still too large
        result: list[tuple[int, int]] = []
        for start, end in pieces:
            piece_len = end - start
            if piece_len <= self._chunk_size:
                result.append((start, end))
            else:
                sub = self._split_text(text[start:end], remaining_seps)
                result.extend((start + s, start + e) for s, e in sub)
        return result

    def _merge_with_overlap(
        self, text: str, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge spans into windows of ≤ chunk_size with chunk_overlap.

        Adjacent spans are accumulated until adding the next one would exceed
        *chunk_size*.  When a window is emitted, the next window starts
        *chunk_overlap* characters before the current window end so context
        is preserved across boundaries.
        """
        if not spans:
            return []

        windows: list[tuple[int, int]] = []
        win_start = spans[0][0]
        win_end = spans[0][1]

        for span_start, span_end in spans[1:]:
            candidate_len = span_end - win_start
            if candidate_len <= self._chunk_size:
                win_end = span_end
            else:
                windows.append((win_start, win_end))
                # Next window begins chunk_overlap chars before current end
                overlap_start = max(win_start, win_end - self._chunk_overlap)
                win_start = overlap_start
                win_end = span_end

        windows.append((win_start, win_end))
        return windows
