"""OpenSearch client wrapper.

Responsibilities
----------------
- Connection management (with retry)
- Index creation with the correct mapping (kNN vector field + metadata fields)
- Search pipeline creation (normalization processor for hybrid search)
- Bulk upsert
- ANN (approximate nearest-neighbour) vector search via native knn query
- BM25 lexical search via native match query
- Hybrid search via OpenSearch native hybrid query + normalization pipeline (RRF)
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

_HYBRID_PIPELINE_ID = "legalrag_hybrid_pipeline"


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

    def _index_embedding_dim_matches(self) -> bool:
        """Return True if existing index has knn_vector embedding with our dimension."""
        try:
            m = self._client.indices.get_mapping(index=self.index_name)
        except Exception:
            return False
        root = next(iter(m.values()), {})
        props = root.get("mappings", {}).get("properties", {})
        emb = props.get("embedding") or {}
        if emb.get("type") != "knn_vector":
            return False
        return int(emb.get("dimension", -1)) == int(self._embedding_dim)

    def ensure_index(self) -> None:
        """Create the index and search pipeline if they do not exist yet.

        If the index already exists but the embedding field is not ``knn_vector``
        or the vector dimension differs (e.g. stale index after a failed DELETE),
        the index is dropped and recreated so ingest/search stay consistent.
        """
        self._ensure_hybrid_pipeline()
        if self._client.indices.exists(index=self.index_name):
            if self._index_embedding_dim_matches():
                logger.debug("Index '%s' already exists with matching knn mapping.", self.index_name)
                return
            logger.warning(
                "Index '%s' exists but embedding mapping mismatches dim=%s; recreating.",
                self.index_name,
                self._embedding_dim,
            )
            self.delete_index()
        mapping = self._build_mapping()
        try:
            self._client.indices.create(index=self.index_name, body=mapping)
            logger.info("Created index '%s'.", self.index_name)
        except RequestError as exc:
            if "resource_already_exists_exception" in str(exc).lower():
                logger.debug("Index '%s' already exists (race condition).", self.index_name)
            else:
                raise

    def _ensure_hybrid_pipeline(self) -> None:
        """Create the hybrid search pipeline if absent.

        Tries pipeline formats in order of preference:
          1. ``score-ranker-processor`` with RRF combination — native RRF, available
             in OpenSearch 2.11+ (renamed from normalization-processor in 3.x).
          2. ``normalization-processor`` with min-max + arithmetic_mean — wider
             compatibility fallback for older OpenSearch 2.x clusters.

        Both approaches combine ANN and BM25 result sets server-side in a single
        request, avoiding the two-request manual RRF used as the final fallback.
        """
        try:
            self._client.transport.perform_request(
                "GET", f"/_search/pipeline/{_HYBRID_PIPELINE_ID}"
            )
            logger.debug("Search pipeline '%s' already exists.", _HYBRID_PIPELINE_ID)
            return
        except Exception:
            pass  # pipeline not found — create it

        # Attempt 1: score-ranker-processor (OpenSearch 2.11+ / 3.x RRF)
        rrf_body = {
            "description": "RRF hybrid pipeline for LegalRAG",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {"technique": "rrf"},
                    }
                }
            ],
        }
        try:
            self._client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{_HYBRID_PIPELINE_ID}",
                body=rrf_body,
            )
            logger.info(
                "Created search pipeline '%s' (score-ranker-processor/RRF).",
                _HYBRID_PIPELINE_ID,
            )
            return
        except Exception as exc:
            logger.debug("score-ranker-processor not available: %s — trying normalization-processor.", exc)

        # Attempt 2: normalization-processor with min-max + arithmetic_mean (OpenSearch 2.x)
        minmax_body = {
            "description": "Hybrid pipeline for LegalRAG (min-max + arithmetic_mean)",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {"technique": "arithmetic_mean"},
                    }
                }
            ],
        }
        try:
            self._client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{_HYBRID_PIPELINE_ID}",
                body=minmax_body,
            )
            logger.info(
                "Created search pipeline '%s' (normalization-processor/min-max).",
                _HYBRID_PIPELINE_ID,
            )
        except Exception as exc:
            logger.warning(
                "Could not create search pipeline '%s': %s. "
                "Hybrid search will fall back to manual RRF.",
                _HYBRID_PIPELINE_ID,
                exc,
            )

    def _build_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                    # ef_search controls recall at query time; 512 is a good balance
                    # for legal text (can be tuned per-query via the knn query params)
                    "knn.algo_param.ef_search": 512,
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
                    # Dense vector for ANN search
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self._embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {"ef_construction": 128, "m": 16},
                        },
                    },
                }
            },
        }

    def delete_index(self) -> None:
        self._client.indices.delete(index=self.index_name, ignore_unavailable=True)
        logger.warning("Deleted index '%s'.", self.index_name)

    # ── Write operations ──────────────────────────────────────────────────────

    # Smaller batches reduce OpenSearch JVM peak memory during large corpus ingests.
    _BULK_CHUNK_SIZE = 400

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def bulk(self, actions: list[dict[str, Any]]) -> None:
        if not actions:
            return
        total_success = 0
        for i in range(0, len(actions), self._BULK_CHUNK_SIZE):
            chunk = actions[i : i + self._BULK_CHUNK_SIZE]
            success, errors = helpers.bulk(self._client, chunk, raise_on_error=False)
            total_success += success
            if errors:
                logger.warning("Bulk index errors (%d): %s", len(errors), errors[:3])
        logger.debug("Bulk indexed %d docs.", total_success)

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
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        """Approximate nearest-neighbour search over child chunks (is_parent=false).

        Uses OpenSearch's native knn query backed by the Lucene HNSW engine.
        ``ef_search`` overrides the index-level default for this request only.
        """
        filter_clauses: list[dict] = [{"term": {"is_parent": False}}]
        if filters:
            filter_clauses.extend(
                {"term": {field: val}} for field, val in filters.items() if val is not None
            )

        knn_clause: dict[str, Any] = {
            "vector": vector,
            "k": k,
            "filter": {"bool": {"must": filter_clauses}},
        }
        if ef_search is not None:
            knn_clause["method_parameters"] = {"ef": ef_search}

        query: dict[str, Any] = {
            "size": k,
            "query": {"knn": {"embedding": knn_clause}},
        }
        resp = self._client.search(index=self.index_name, body=query)
        return resp["hits"]["hits"]

    def bm25_search(
        self,
        query_text: str,
        k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 full-text search over child chunks using OpenSearch's native match query."""
        must: list[dict] = [
            {"match": {"text": {"query": query_text}}},
            {"term": {"is_parent": False}},
        ]
        filter_clauses: list[dict] = []
        if filters:
            filter_clauses = [
                {"term": {field: val}} for field, val in filters.items() if val is not None
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
    ) -> list[dict[str, Any]]:
        """Hybrid search using OpenSearch's native hybrid query + RRF normalization pipeline.

        Sends a single request with both a knn sub-query and a BM25 match sub-query.
        The ``legalrag_hybrid_pipeline`` search pipeline combines the two ranked lists
        using Reciprocal Rank Fusion before returning results.

        Falls back to manual RRF if the pipeline is unavailable.
        """
        filter_clauses: list[dict] = [{"term": {"is_parent": False}}]
        if filters:
            filter_clauses.extend(
                {"term": {field: val}} for field, val in filters.items() if val is not None
            )
        bool_filter = {"bool": {"must": filter_clauses}}

        query: dict[str, Any] = {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        # Sub-query 1: ANN vector search
                        {
                            "knn": {
                                "embedding": {
                                    "vector": vector,
                                    "k": k,
                                    "filter": bool_filter,
                                }
                            }
                        },
                        # Sub-query 2: BM25 lexical search
                        {
                            "bool": {
                                "must": [{"match": {"text": {"query": query_text}}}],
                                "filter": filter_clauses,
                            }
                        },
                    ]
                }
            },
        }

        try:
            resp = self._client.search(
                index=self.index_name,
                body=query,
                params={"search_pipeline": _HYBRID_PIPELINE_ID},
            )
            return resp["hits"]["hits"]
        except Exception as exc:
            logger.warning(
                "Native hybrid search failed (%s); falling back to manual RRF.", exc
            )
            try:
                semantic_hits = self.knn_search(vector, k=k, filters=filters)
            except Exception as knn_exc:
                logger.warning(
                    "kNN fallback failed (%s); using BM25-only retrieval for this query.",
                    knn_exc,
                )
                return self.bm25_search(query_text, k=k, filters=filters)
            lexical_hits = self.bm25_search(query_text, k=k, filters=filters)
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


# ── Fallback utility ──────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    list_a: list[dict], list_b: list[dict], k: int = 60
) -> list[dict]:
    """Manual RRF fallback used only when the native hybrid pipeline is unavailable.

    Each document gets score = Σ 1/(rank + k) across all lists; k=60 is standard.
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
