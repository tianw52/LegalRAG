"""
Abstract base classes (interfaces) for every swappable component.

Design principle: concrete implementations live in their own modules and
receive settings via constructor injection.  No module imports a concrete
implementation from another pipeline stage – only these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from legalrag.core.models import (
    Chunk,
    RAGResponse,
    RawDocument,
    RetrievedChunk,
    StructuredQuery,
)


# ── Ingestion interfaces ──────────────────────────────────────────────────────


class BaseLoader(ABC):
    """Loads raw documents from a source (file path, directory, S3, …)."""

    @abstractmethod
    def load(self, source: str) -> list[RawDocument]:
        """Return a list of RawDocuments from *source*."""


class BaseChunker(ABC):
    """Splits a RawDocument into Chunks."""

    @abstractmethod
    def chunk(self, document: RawDocument) -> list[Chunk]:
        """Return ordered list of Chunks for *document*."""

    @property
    def is_hierarchical(self) -> bool:
        """True if this chunker produces parent+child pairs.

        When False (flat chunkers), all produced chunks should be embedded
        and retrieved directly — none are treated as parents.
        Subclasses override this; default is False.
        """
        return False


class BaseMetadataExtractor(ABC):
    """Enriches chunk metadata using heuristics or an LLM."""

    @abstractmethod
    def extract(self, document: RawDocument) -> RawDocument:
        """Return *document* with its metadata fields populated."""


class BaseEmbedder(ABC):
    """Encodes text into dense vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""


class BaseIndexer(ABC):
    """Writes chunks (with embeddings) to the vector store."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Upsert *chunks* into the index."""

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove all chunks belonging to *doc_id*."""


# ── Query interfaces ──────────────────────────────────────────────────────────


class BaseQueryFormulator(ABC):
    """Transforms a raw user query into a StructuredQuery."""

    @abstractmethod
    def formulate(self, raw_query: str) -> StructuredQuery:
        """Return a structured, enriched query."""


class BaseRetriever(ABC):
    """Retrieves candidate chunks from the vector store."""

    @abstractmethod
    def retrieve(self, query: StructuredQuery) -> list[RetrievedChunk]:
        """Return top-K candidates ranked by retrieval score."""


class BaseReranker(ABC):
    """Re-scores a candidate list and returns the top-N most relevant chunks."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Return *top_n* chunks sorted by rerank score descending."""


class BaseRouter(ABC):
    """
    Decides which downstream path to follow based on the query and candidates.
    Paths: 'reranker' (default fast path) | 'deep_search' (RLM-style, slow path).
    """

    @abstractmethod
    def route(
        self, query: StructuredQuery, candidates: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Return (path_name, final_chunks).

        path_name is 'reranker' or 'deep_search'.
        """


class BaseGenerator(ABC):
    """Generates the final answer conditioned on retrieved context."""

    @abstractmethod
    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> RAGResponse:
        """Return a RAGResponse with the answer and provenance."""

    @abstractmethod
    def stream(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Yield answer tokens one by one (streaming variant)."""
