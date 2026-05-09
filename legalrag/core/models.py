"""Shared domain models (Pydantic schemas) used across ingestion and query."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def stable_id(*parts: str) -> str:
    """Return a deterministic hex ID from one or more string parts.

    Used to make doc_id and chunk_id stable across ingestion runs so that
    re-ingesting the same file overwrites existing documents (upsert) instead
    of creating duplicates.
    """
    key = "|".join(parts)
    return hashlib.md5(key.encode()).hexdigest()


def doc_id_from_citation(citation: str | None, source_path: str) -> str:
    """Derive a stable doc_id.

    Priority:
      1. citation  (e.g. "2010 BCCA 220") – content-based, path-independent
      2. filename stem  (e.g. "2010_BCCA_2010 BCCA 220 Harrison...") – fallback
         for the ~0.7% of documents without a parseable citation

    This means the ID survives folder renames, path changes, and re-downloads
    of the same document to a different location.
    """
    if citation:
        return stable_id("citation", citation)
    return stable_id("stem", Path(source_path).stem)


# ── Document models ───────────────────────────────────────────────────────────


class LegalDocumentMetadata(BaseModel):
    """Metadata fields extracted from / assigned to a legal document."""

    # Set automatically on construction from source_path (path-based fallback).
    # The ingestion pipeline overwrites this with doc_id_from_citation() once
    # the metadata extractor has populated the citation field.
    doc_id: str = Field(default="")
    source_path: str
    doc_type: str | None = None          # e.g. "case", "statute", "regulation"
    court: str | None = None             # full name e.g. "British Columbia Court of Appeal"
    court_abbrev: str | None = None      # short code e.g. "BCCA"
    citation: str | None = None          # neutral citation e.g. "2010 BCCA 220"
    case_name: str | None = None         # e.g. "Harrison v. British Columbia ..."
    decision_date: date | None = None
    year: int | None = None
    pages: int | None = None
    url: str | None = None               # CanLII source URL
    jurisdiction: str | None = None      # for future use
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_doc_id(self) -> "LegalDocumentMetadata":
        """Set a path-based doc_id at construction time if none was provided.

        This is the construction-time fallback only. The ingestion pipeline
        calls doc_id_from_citation() after metadata extraction to upgrade the
        ID to a citation-based one.
        """
        if not self.doc_id:
            self.doc_id = stable_id("stem", Path(self.source_path).stem)
        return self


class RawDocument(BaseModel):
    """A document as loaded from disk – plain text + metadata."""

    metadata: LegalDocumentMetadata
    text: str


# ── Chunk models ──────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """An individual text chunk with its lineage and embedding."""

    # The chunker always supplies a stable_id()-derived chunk_id.
    # The random uuid4 fallback is only for ad-hoc construction in tests.
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str                          # foreign key → LegalDocumentMetadata.doc_id
    parent_chunk_id: str | None = None   # for hierarchical chunking: parent summary chunk
    is_parent: bool = False              # True only for HierarchicalChunker parent chunks
    text: str
    char_start: int | None = None
    char_end: int | None = None
    # None only during intermediate construction in the chunker; always set before indexing
    metadata: LegalDocumentMetadata | None = None
    embedding: list[float] | None = None


# ── Query / retrieval models ──────────────────────────────────────────────────


class StructuredQuery(BaseModel):
    """
    Structured representation of the user query after LLM-based query formulation.
    Used to drive both metadata filtering and embedding retrieval.
    """

    raw_query: str
    reformulated_query: str             # cleaned / expanded query for embedding
    lexical_keywords: list[str] = Field(default_factory=list)

    # Metadata filters (only populated when the LLM extracts them)
    court_filter: str | None = None
    citation_filter: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, annotated with scores."""

    chunk: Chunk
    semantic_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None


class RAGResponse(BaseModel):
    """Final answer object returned to the caller."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    router_path: str = ""               # "reranker" | "deep_search"
    metadata: dict[str, Any] = Field(default_factory=dict)
