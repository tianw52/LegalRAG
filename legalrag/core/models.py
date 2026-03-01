"""Shared domain models (Pydantic schemas) used across ingestion and query."""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Document models ───────────────────────────────────────────────────────────


class LegalDocumentMetadata(BaseModel):
    """Metadata fields extracted from / assigned to a legal document."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
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


class RawDocument(BaseModel):
    """A document as loaded from disk – plain text + metadata."""

    metadata: LegalDocumentMetadata
    text: str


# ── Chunk models ──────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """An individual text chunk with its lineage and embedding."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str                          # foreign key → LegalDocumentMetadata.doc_id
    parent_chunk_id: str | None = None   # for hierarchical chunking: parent summary chunk
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
