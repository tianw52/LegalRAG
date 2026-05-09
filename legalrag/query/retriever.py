"""Retriever: fetches candidate chunks from OpenSearch.

Supports three modes:
  - semantic   : kNN vector search only
  - lexical    : BM25 text search only
  - hybrid     : Reciprocal Rank Fusion of both (default)

The retriever also handles metadata filtering (court, citation, date range)
by translating StructuredQuery filter fields into OpenSearch filter clauses.
"""

from __future__ import annotations

import logging
from typing import Literal

from legalrag.core.interfaces import BaseEmbedder, BaseRetriever
from legalrag.core.models import Chunk, LegalDocumentMetadata, RetrievedChunk, StructuredQuery
from legalrag.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

RetrievalMode = Literal["semantic", "lexical", "hybrid"]


class OpenSearchRetriever(BaseRetriever):
    """Retrieves candidate chunks using OpenSearch kNN and/or BM25."""

    def __init__(
        self,
        client: OpenSearchClient,
        embedder: BaseEmbedder,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
    ) -> None:
        from legalrag.core.config import settings

        self._client = client
        self._embedder = embedder
        self._mode = mode
        self._top_k = top_k or settings.retrieval.top_k

    def retrieve(self, query: StructuredQuery) -> list[RetrievedChunk]:
        filters = self._build_filters(query)
        query_text = query.reformulated_query or query.raw_query

        if self._mode == "semantic":
            vector = self._embedder.embed([query_text])[0]
            hits = self._client.knn_search(vector, k=self._top_k, filters=filters)
            return [self._hit_to_retrieved(h, semantic_score=h["_score"]) for h in hits]

        if self._mode == "lexical":
            hits = self._client.bm25_search(query_text, k=self._top_k, filters=filters)
            return [self._hit_to_retrieved(h, lexical_score=h["_score"]) for h in hits]

        # hybrid (default) — native OpenSearch hybrid query with RRF pipeline
        vector = self._embedder.embed([query_text])[0]
        hits = self._client.hybrid_search(
            vector, query_text, k=self._top_k, filters=filters
        )
        return [
            self._hit_to_retrieved(
                h,
                # _score from the native pipeline is the combined RRF score;
                # expose it as semantic_score so downstream rerankers can use it
                semantic_score=h.get("_score") or h.get("_rrf_score"),
            )
            for h in hits
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_filters(self, query: StructuredQuery) -> dict[str, str | None]:
        """Translate StructuredQuery metadata filters to OpenSearch terms."""
        filters: dict[str, str | None] = {}
        if query.court_filter:
            filters["court"] = query.court_filter
        if query.citation_filter:
            filters["citation"] = query.citation_filter
        # Date range handled separately if needed via range query; placeholder for now
        return filters

    @staticmethod
    def _hit_to_retrieved(
        hit: dict,
        *,
        semantic_score: float | None = None,
        lexical_score: float | None = None,
    ) -> RetrievedChunk:
        src = hit["_source"]
        meta = LegalDocumentMetadata(
            doc_id=src.get("doc_id", ""),
            source_path=src.get("source_path", ""),
            court=src.get("court"),
            citation=src.get("citation"),
        )
        chunk = Chunk(
            chunk_id=src.get("chunk_id", hit["_id"]),
            doc_id=src.get("doc_id", ""),
            parent_chunk_id=src.get("parent_chunk_id"),
            text=src.get("text", ""),
            char_start=src.get("char_start"),
            char_end=src.get("char_end"),
            metadata=meta,
        )
        return RetrievedChunk(
            chunk=chunk,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
        )
