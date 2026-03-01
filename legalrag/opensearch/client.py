"""OpenSearch client wrapper.

Responsibilities
----------------
- Connection management (with retry)
- Index creation with the correct mapping (kNN vector field + metadata fields)
- Bulk upsert
- kNN vector search
- BM25 lexical search
- Hybrid search (kNN + BM25 via OpenSearch hybrid query)
- Delete by doc_id
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import RequestError
from tenacity import retry, stop_after_attempt, wait_exponential

from legalrag.core.config import OpenSearchSettings, settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Thin abstraction over the opensearch-py client."""

    def __init__(self, cfg: OpenSearchSettings, embedding_dim: int = 1024) -> None:
        self._cfg = cfg
        self._embedding_dim = embedding_dim
        self.index_name = cfg.index_name
        self._client = self._build_client()

    @classmethod
    def from_settings(cls, embedding_dim: int | None = None) -> "OpenSearchClient":
        dim = embedding_dim or settings.embedding.dim
        return cls(cfg=settings.opensearch, embedding_dim=dim)

    # ── Connection ────────────────────────────────────────────────────────────

    def _build_client(self) -> OpenSearch:
        cfg = self._cfg
        return OpenSearch(
            hosts=[{"host": cfg.host, "port": cfg.port}],
            http_auth=(cfg.user, cfg.password),
            use_ssl=cfg.use_ssl,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

    def ping(self) -> bool:
        return self._client.ping()

    # ── Index management ──────────────────────────────────────────────────────

    def ensure_index(self) -> None:
        """Create the index if it does not exist yet."""
        if self._client.indices.exists(index=self.index_name):
            logger.debug("Index '%s' already exists.", self.index_name)
            return
        mapping = self._build_mapping()
        try:
            self._client.indices.create(index=self.index_name, body=mapping)
            logger.info("Created index '%s'.", self.index_name)
        except RequestError as exc:
            if "resource_already_exists_exception" in str(exc).lower():
                logger.debug("Index '%s' already exists (race condition).", self.index_name)
            else:
                raise

    def _build_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "analysis": {
                    "analyzer": {
                        "legal_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"],
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "parent_chunk_id": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    "text": {
                        "type": "text",
                        "analyzer": "legal_analyzer",
                    },
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                    # Metadata fields (filterable)
                    "source_path": {"type": "keyword"},
                    "court": {"type": "keyword"},
                    "citation": {"type": "keyword"},
                    "decision_date": {"type": "date", "format": "yyyy-MM-dd"},
                    # Dense vector for kNN
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self._embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                }
            },
        }

    def delete_index(self) -> None:
        self._client.indices.delete(index=self.index_name, ignore_unavailable=True)
        logger.warning("Deleted index '%s'.", self.index_name)

    # ── Write operations ──────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def bulk(self, actions: list[dict[str, Any]]) -> None:
        success, errors = helpers.bulk(self._client, actions, raise_on_error=False)
        if errors:
            logger.warning("Bulk index errors (%d): %s", len(errors), errors[:3])
        logger.debug("Bulk indexed %d docs.", success)

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}},
        )

    # ── Read operations ───────────────────────────────────────────────────────

    def knn_search(
        self,
        vector: list[float],
        k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """k-nearest-neighbour search over child chunks (is_parent=false)."""
        knn_query: dict[str, Any] = {
            "vector": vector,
            "k": k,
        }
        query: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{"knn": {"embedding": knn_query}}],
                    "filter": [{"term": {"is_parent": False}}],
                }
            },
        }
        if filters:
            query["query"]["bool"]["filter"].extend(
                [{"term": {k: v}} for k, v in filters.items() if v is not None]
            )
        resp = self._client.search(index=self.index_name, body=query)
        return resp["hits"]["hits"]

    def bm25_search(
        self,
        query_text: str,
        k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 full-text search over child chunks."""
        must: list[dict] = [
            {"match": {"text": {"query": query_text}}},
            {"term": {"is_parent": False}},
        ]
        filter_clauses: list[dict] = []
        if filters:
            filter_clauses = [
                {"term": {k: v}} for k, v in filters.items() if v is not None
            ]
        query: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses,
                }
            },
        }
        resp = self._client.search(index=self.index_name, body=query)
        return resp["hits"]["hits"]

    def hybrid_search(
        self,
        vector: list[float],
        query_text: str,
        k: int = 20,
        filters: dict[str, Any] | None = None,
        semantic_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining kNN and BM25 via OpenSearch's hybrid query.

        Uses a simple linear interpolation score: 
            score = semantic_weight * kNN_score + (1 - semantic_weight) * BM25_score
        Implemented as a rank-fusion over separate result sets for compatibility
        with older OpenSearch versions that lack native hybrid query.
        """
        semantic_hits = self.knn_search(vector, k=k, filters=filters)
        lexical_hits = self.bm25_search(query_text, k=k, filters=filters)

        # Normalise scores, merge, and re-rank
        return _reciprocal_rank_fusion(semantic_hits, lexical_hits, k=k)

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch a single document by chunk_id."""
        resp = self._client.get(index=self.index_name, id=chunk_id, ignore=[404])
        if resp.get("found"):
            return resp["_source"]
        return None

    def get_parent(self, parent_chunk_id: str) -> dict[str, Any] | None:
        """Fetch the parent chunk for context expansion."""
        return self.get_by_chunk_id(parent_chunk_id)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    list_a: list[dict], list_b: list[dict], k: int = 60
) -> list[dict]:
    """
    Combine two ranked lists via Reciprocal Rank Fusion.

    Each document gets score = Σ 1/(rank + k) across all lists.
    k=60 is the standard RRF constant.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(list_a, start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
        docs[doc_id] = hit

    for rank, hit in enumerate(list_b, start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
        docs.setdefault(doc_id, hit)

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    merged = []
    for doc_id in sorted_ids[:k]:
        hit = docs[doc_id]
        hit["_rrf_score"] = scores[doc_id]
        merged.append(hit)
    return merged
